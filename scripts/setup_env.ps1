#!/usr/bin/env pwsh
<#
.SYNOPSIS
	Setup build environment for Ray Tracer
.DESCRIPTION
	Automatically detects and sets CUDA and OptiX paths
#>

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Ray Tracer Build Environment Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Detect CUDA Toolkit
$cudaBase = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
$cudaVersions = @("v13.3", "v13.2", "v12.6", "v12.5")

$cudaFound = $false
foreach ($version in $cudaVersions) {
	$cudaPath = Join-Path $cudaBase $version
	if (Test-Path $cudaPath) {
		$env:CudaToolkitPath = $cudaPath
		Write-Host "[OK] Found CUDA Toolkit $version" -ForegroundColor Green
		$cudaFound = $true
		break
	}
}

if (-not $cudaFound) {
	Write-Host "[WARNING] CUDA Toolkit not found!" -ForegroundColor Yellow
	Write-Host "Please install CUDA Toolkit 12.x or 13.x" -ForegroundColor Yellow
}

# Detect OptiX SDK
$optixBase = "C:\ProgramData\NVIDIA Corporation"
$optixVersions = @("OptiX SDK 9.1.0", "OptiX SDK 9.0.0")

$optixFound = $false
foreach ($version in $optixVersions) {
	$optixPath = Join-Path $optixBase $version
	if (Test-Path $optixPath) {
		$env:OptixSdkPath = $optixPath
		Write-Host "[OK] Found $version" -ForegroundColor Green
		$optixFound = $true
		break
	}
}

if (-not $optixFound) {
	Write-Host "[WARNING] OptiX SDK not found!" -ForegroundColor Yellow
	Write-Host "Please install OptiX SDK 9.1+" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Environment variables set for this session:" -ForegroundColor Cyan
Write-Host "  CudaToolkitPath = $env:CudaToolkitPath" -ForegroundColor Gray
Write-Host "  OptixSdkPath    = $env:OptixSdkPath" -ForegroundColor Gray
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Ready to build!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Run: .\build_all.ps1" -ForegroundColor Yellow
Write-Host "Or:  .\build_all.bat" -ForegroundColor Yellow
Write-Host ""
