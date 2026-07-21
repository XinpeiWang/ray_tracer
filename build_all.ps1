#!/usr/bin/env pwsh
<#
.SYNOPSIS
	Build all components of the Ray Tracer project
.DESCRIPTION
	This script builds:
	1. C++ renderer components (CPU + OptiX GPU) via MSBuild
	2. Tests project
	3. Qt GUI application (optional)
.PARAMETER Configuration
	Build configuration: Debug or Release (default: Release)
.PARAMETER SkipTests
	Skip building tests
.PARAMETER SkipGui
	Skip building Qt GUI
.PARAMETER Deploy
	Deploy Qt GUI after building
.EXAMPLE
	.\build_all.ps1
	.\build_all.ps1 -Configuration Debug
	.\build_all.ps1 -SkipGui
	.\build_all.ps1 -Deploy
#>

param(
	[ValidateSet("Debug", "Release")]
	[string]$Configuration = "Release",

	[switch]$SkipTests,
	[switch]$SkipGui,
	[switch]$Deploy,
	[switch]$Clean
)

$ErrorActionPreference = "Stop"
$script:BuildFailed = $false

function Write-Header {
	param([string]$Message)
	Write-Host ""
	Write-Host "========================================" -ForegroundColor Cyan
	Write-Host $Message -ForegroundColor Cyan
	Write-Host "========================================" -ForegroundColor Cyan
}

function Write-Success {
	param([string]$Message)
	Write-Host "✓ $Message" -ForegroundColor Green
}

function Write-Error-Message {
	param([string]$Message)
	Write-Host "✗ $Message" -ForegroundColor Red
	$script:BuildFailed = $true
}

function Write-Warning-Message {
	param([string]$Message)
	Write-Host "⚠ $Message" -ForegroundColor Yellow
}

# Check prerequisites
Write-Header "Checking Prerequisites"

# Check MSBuild
try {
	$msbuild = Get-Command msbuild -ErrorAction Stop
	Write-Success "MSBuild found: $($msbuild.Source)"
} catch {
	Write-Error-Message "MSBuild not found. Please run this from a Visual Studio Developer Command Prompt or Developer PowerShell."
	exit 1
}

# Check CUDA Toolkit (required for OptiX)
if (-not $env:CudaToolkitPath) {
	Write-Warning-Message "CudaToolkitPath not set. OptiX build may fail."
} else {
	Write-Success "CUDA Toolkit: $env:CudaToolkitPath"
}

# Check OptiX SDK (required for GPU rendering)
if (-not $env:OptixSdkPath) {
	Write-Warning-Message "OptixSdkPath not set. OptiX build may fail."
} else {
	Write-Success "OptiX SDK: $env:OptixSdkPath"
}

# Clean if requested
if ($Clean) {
	Write-Header "Cleaning Previous Build"
	msbuild ray_tracer.sln /t:Clean /p:Configuration=$Configuration /p:Platform=x64 /v:minimal
	if ($LASTEXITCODE -eq 0) {
		Write-Success "Clean completed"
	} else {
		Write-Warning-Message "Clean had warnings"
	}
}

# Build C++ solution
Write-Header "Building C++ Solution ($Configuration|x64)"
Write-Host "Building: launcher, cpu_renderer, optix_renderer$(if (-not $SkipTests) {', tests'})"

if ($SkipTests) {
	# Build only the main projects
	msbuild launcher/launcher.vcxproj /p:Configuration=$Configuration /p:Platform=x64 /v:minimal /m
} else {
	# Build entire solution including tests
	msbuild ray_tracer.sln /p:Configuration=$Configuration /p:Platform=x64 /v:minimal /m
}

if ($LASTEXITCODE -ne 0) {
	Write-Error-Message "C++ build failed"
	exit 1
}

Write-Success "C++ build completed"

# Verify outputs
Write-Header "Verifying Build Outputs"

$exePath = "launcher\x64\$Configuration\ray_tracer.exe"
if (Test-Path $exePath) {
	$size = (Get-Item $exePath).Length / 1MB
	Write-Success "Launcher: $exePath ($([math]::Round($size, 2)) MB)"
} else {
	Write-Error-Message "Launcher executable not found: $exePath"
}

$cpuLib = "cpu_renderer\x64\$Configuration\cpu_renderer.lib"
if (Test-Path $cpuLib) {
	Write-Success "CPU Renderer: $cpuLib"
} else {
	Write-Error-Message "CPU renderer library not found: $cpuLib"
}

$optixLib = "x64\$Configuration\optix_renderer.lib"
if (Test-Path $optixLib) {
	Write-Success "OptiX Renderer: $optixLib"
} else {
	Write-Error-Message "OptiX renderer library not found: $optixLib"
}

$ptxFile = "gpu\optix\optix_programs.ptx"
if (Test-Path $ptxFile) {
	Write-Success "OptiX PTX: $ptxFile"
} else {
	Write-Error-Message "OptiX PTX not found: $ptxFile"
}

if (-not $SkipTests) {
	$testsExe = "tests\x64\$Configuration\ray_tracer_tests.exe"
	if (Test-Path $testsExe) {
		Write-Success "Tests: $testsExe"
	} else {
		Write-Warning-Message "Tests executable not found: $testsExe"
	}
}

# Build Qt GUI
if (-not $SkipGui) {
	Write-Header "Building Qt GUI"

	# Check if Qt is available
	$qmake = Get-Command qmake -ErrorAction SilentlyContinue
	if (-not $qmake) {
		Write-Warning-Message "qmake not found. Skipping Qt GUI build."
		Write-Host "To build Qt GUI, ensure Qt is in PATH or run from Qt command prompt."
	} else {
		Push-Location qt_gui
		try {
			# Clean old build
			if (Test-Path Makefile) {
				if ($Clean) {
					& $qmake.Source -r "CONFIG+=$($Configuration.ToLower())"
					nmake clean 2>$null
				}
			}

			# Generate makefiles
			& $qmake.Source -r "CONFIG+=$($Configuration.ToLower())"
			if ($LASTEXITCODE -ne 0) {
				Write-Error-Message "qmake failed"
			} else {
				# Build
				nmake
				if ($LASTEXITCODE -ne 0) {
					Write-Error-Message "Qt GUI build failed"
				} else {
					Write-Success "Qt GUI build completed"

					# Check output
					$guiExe = "$($Configuration.ToLower())\RayTracerGUI.exe"
					if (Test-Path $guiExe) {
						Write-Success "Qt GUI: qt_gui\$guiExe"
					} else {
						Write-Warning-Message "Qt GUI executable not found"
					}
				}
			}
		} finally {
			Pop-Location
		}
	}
}

# Deploy if requested
if ($Deploy -and -not $SkipGui) {
	Write-Header "Deploying Qt GUI Package"

	if (Test-Path "deploy_qt_gui.ps1") {
		.\deploy_qt_gui.ps1 -Configuration $Configuration
		if ($LASTEXITCODE -eq 0) {
			Write-Success "Deployment completed"
		} else {
			Write-Warning-Message "Deployment had issues"
		}
	} else {
		Write-Warning-Message "deploy_qt_gui.ps1 not found. Skipping deployment."
	}
}

# Summary
Write-Header "Build Summary"

if ($script:BuildFailed) {
	Write-Host "Build completed with errors" -ForegroundColor Red
	exit 1
} else {
	Write-Host "All builds completed successfully!" -ForegroundColor Green
	Write-Host ""
	Write-Host "Next steps:" -ForegroundColor Cyan
	Write-Host "  - Run launcher:    .\launcher\x64\$Configuration\ray_tracer.exe --help"
	if (-not $SkipTests) {
		Write-Host "  - Run tests:       .\tests\x64\$Configuration\ray_tracer_tests.exe"
	}
	if (-not $SkipGui) {
		Write-Host "  - Run Qt GUI:      .\qt_gui\$($Configuration.ToLower())\RayTracerGUI.exe"
		if ($Deploy) {
			Write-Host "  - Run deployed:    .\RayTracer_Package\RayTracerGUI.exe"
		}
	}
}
