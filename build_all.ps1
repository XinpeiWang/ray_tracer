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
	Write-Error "Launcher executable not found: $exePath"
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

	# Find Qt installation
	$qtPaths = @(
		"C:\Qt\6.11.1\mingw_64\bin",
		"C:\Qt\6.10.0\mingw_64\bin",
		"C:\Qt\6.9.0\mingw_64\bin"
	)

	$qtBinPath = $null
	foreach ($path in $qtPaths) {
		if (Test-Path "$path\qmake.exe") {
			$qtBinPath = $path
			break
		}
	}

	# Check if Qt is available
	$qmake = Get-Command qmake -ErrorAction SilentlyContinue
	if (-not $qmake -and -not $qtBinPath) {
		Write-Warning-Message "Qt not found in PATH or common locations. Skipping Qt GUI build."
		Write-Host "To build Qt GUI, ensure Qt is in PATH or install Qt to C:\Qt\"
	} else {
		if (-not $qmake) {
			$qmake = Get-Command "$qtBinPath\qmake.exe"
		}

		# Find MinGW
		$mingwPaths = @(
			"C:\Qt\Tools\mingw1310_64\bin",
			"C:\Qt\Tools\mingw1120_64\bin",
			"C:\Qt\Tools\mingw_64\bin"
		)

		$mingwBinPath = $null
		foreach ($path in $mingwPaths) {
			if (Test-Path "$path\mingw32-make.exe") {
				$mingwBinPath = $path
				break
			}
		}

		if (-not $mingwBinPath) {
			Write-Warning-Message "MinGW not found. Skipping Qt GUI build."
			Write-Host "Expected MinGW at C:\Qt\Tools\mingw*_64\bin\"
		} else {
			Write-Success "Qt: $($qmake.Source)"
			Write-Success "MinGW: $mingwBinPath"

			Push-Location qt_gui
			try {
				# Add Qt and MinGW to PATH for this session
				$env:PATH = "$qtBinPath;$mingwBinPath;$env:PATH"

				# Clean old build
				if (Test-Path Makefile) {
					if ($Clean) {
						& "$mingwBinPath\mingw32-make.exe" clean 2>$null
					}
				}

				# Generate makefiles
				& $qmake.Source "RayTracerGUI.pro" -spec win32-g++ "CONFIG+=$($Configuration.ToLower())"
				if ($LASTEXITCODE -ne 0) {
					Write-Error-Message "qmake failed"
				} else {
					# Build
					& "$mingwBinPath\mingw32-make.exe" -j8
					if ($LASTEXITCODE -ne 0) {
						Write-Error-Message "Qt GUI build failed"
					} else {
						Write-Success "Qt GUI build completed"

						# Check output (Qt builds directly to ../RayTracer_Package)
						$guiExe = "..\RayTracer_Package\RayTracerGUI.exe"
						if (Test-Path $guiExe) {
							Write-Success "Qt GUI: $guiExe"
						} else {
							Write-Warning-Message "Qt GUI executable not found at expected location"
						}
					}
				}
			} finally {
				Pop-Location
			}
		}
	}
}

# Deploy if requested
if ($Deploy -and -not $SkipGui) {
	Write-Header "Deploying Qt GUI Package"

	if (Test-Path "deploy_qt_gui.ps1") {
		# Run deployment script with Configuration parameter
		$deployArgs = @("-Configuration", $Configuration)
		& "$PSScriptRoot\deploy_qt_gui.ps1" @deployArgs

		if ($LASTEXITCODE -eq 0) {
			Write-Success "Deployment completed"
		} else {
			Write-Warning-Message "Deployment had issues (exit code: $LASTEXITCODE)"
		}
	} else {
		Write-Warning-Message "deploy_qt_gui.ps1 not found. Skipping deployment."
	}
} elseif (-not $SkipGui -and -not $Deploy) {
	Write-Host ""
	Write-Host "Tip: Use -Deploy flag to automatically package Qt GUI with all dependencies" -ForegroundColor Cyan
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

	# Find launcher path
	$launcherPath = $null
	if (Test-Path "x64\$Configuration\ray_tracer.exe") {
		$launcherPath = "x64\$Configuration\ray_tracer.exe"
	} elseif (Test-Path "launcher\x64\$Configuration\ray_tracer.exe") {
		$launcherPath = "launcher\x64\$Configuration\ray_tracer.exe"
	}

	if ($launcherPath) {
		Write-Host "  - Run launcher:    .\$launcherPath --help"
	}

	if (-not $SkipTests) {
		Write-Host "  - Run tests:       .\tests\x64\$Configuration\ray_tracer_tests.exe"
	}
	if (-not $SkipGui) {
		if ($Deploy) {
			Write-Host "  - Run GUI:         .\RayTracer_Package\RayTracerGUI.exe" -ForegroundColor Green
			Write-Host ""
			Write-Host "Ready-to-run package deployed to: .\RayTracer_Package\" -ForegroundColor Green
		} else {
			Write-Host "  - Run Qt GUI:      .\qt_gui\$($Configuration.ToLower())\RayTracerGUI.exe"
			Write-Host ""
			Write-Host "Tip: Use '.\build_all.ps1 -Deploy' to create a complete package with all dependencies" -ForegroundColor Yellow
		}
	}
}
