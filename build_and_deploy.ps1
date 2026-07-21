#!/usr/bin/env pwsh
<#
.SYNOPSIS
	One-command build and deploy for Ray Tracer project
.DESCRIPTION
	This convenience script:
	1. Builds all C++ components (CPU + OptiX GPU renderer + Launcher)
	2. Builds Qt GUI application
	3. Deploys everything to RayTracer_Package with all dependencies
	4. Validates the package is ready to run
.PARAMETER Configuration
	Build configuration: Debug or Release (default: Release)
.PARAMETER SkipTests
	Skip building tests
.PARAMETER Clean
	Clean before building
.EXAMPLE
	.\build_and_deploy.ps1
	.\build_and_deploy.ps1 -Configuration Debug
	.\build_and_deploy.ps1 -Clean
#>

param(
	[ValidateSet("Debug", "Release")]
	[string]$Configuration = "Release",

	[switch]$SkipTests,
	[switch]$Clean
)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "╔════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║     Ray Tracer - One-Command Build & Deploy Script         ║" -ForegroundColor Cyan
Write-Host "╠════════════════════════════════════════════════════════════╣" -ForegroundColor Cyan
Write-Host "║  Configuration: $Configuration" -ForegroundColor Cyan
Write-Host "║  This will build and package everything for you!            ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

$buildArgs = @(
	"-Configuration", $Configuration,
	"-Deploy"
)

if ($SkipTests) {
	$buildArgs += "-SkipTests"
}

if ($Clean) {
	$buildArgs += "-Clean"
}

try {
	# Run the main build script with deployment enabled
	& "$PSScriptRoot\build_all.ps1" @buildArgs

	if ($LASTEXITCODE -eq 0) {
		Write-Host ""
		Write-Host "╔════════════════════════════════════════════════════════════╗" -ForegroundColor Green
		Write-Host "║               BUILD & DEPLOY SUCCESSFUL!                    ║" -ForegroundColor Green
		Write-Host "╚════════════════════════════════════════════════════════════╝" -ForegroundColor Green
		Write-Host ""
		Write-Host "Your ready-to-run package is in: .\RayTracer_Package\" -ForegroundColor Green
		Write-Host ""
		Write-Host "To launch the GUI:" -ForegroundColor Cyan
		Write-Host "  .\RayTracer_Package\RayTracerGUI.exe" -ForegroundColor White
		Write-Host ""
		Write-Host "Or from PowerShell:" -ForegroundColor Cyan
		Write-Host "  Start-Process '.\RayTracer_Package\RayTracerGUI.exe'" -ForegroundColor White
		Write-Host ""

		# Quick test
		Write-Host "Quick validation:" -ForegroundColor Yellow
		$packageFiles = @(
			"RayTracer_Package\RayTracerGUI.exe",
			"RayTracer_Package\ray_tracer.exe",
			"RayTracer_Package\optix_programs.ptx",
			"RayTracer_Package\Qt6Core.dll",
			"RayTracer_Package\Qt6Gui.dll",
			"RayTracer_Package\Qt6Widgets.dll"
		)

		$allPresent = $true
		foreach ($file in $packageFiles) {
			$exists = Test-Path $file
			if ($exists) {
				$size = (Get-Item $file).Length
				$sizeStr = if ($size -gt 1MB) { "$([math]::Round($size/1MB, 1)) MB" } else { "$([math]::Round($size/1KB)) KB" }
				Write-Host "  ✓ $(Split-Path -Leaf $file) ($sizeStr)" -ForegroundColor Green
			} else {
				Write-Host "  ✗ $(Split-Path -Leaf $file) - MISSING" -ForegroundColor Red
				$allPresent = $false
			}
		}

		Write-Host ""
		if ($allPresent) {
			Write-Host "All essential files present! Package is ready to use. 🎉" -ForegroundColor Green
		} else {
			Write-Host "Some files are missing. Review the build output above." -ForegroundColor Yellow
		}

	} else {
		Write-Host ""
		Write-Host "Build failed. Please review the errors above." -ForegroundColor Red
		exit 1
	}
} catch {
	Write-Host ""
	Write-Host "Build script failed with error: $_" -ForegroundColor Red
	exit 1
}
