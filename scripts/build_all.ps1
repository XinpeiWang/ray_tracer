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

	[switch]$SkipTests = $true,   # Tests require CMake setup; skip by default
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
	Write-Host "[OK] $Message" -ForegroundColor Green
}

function Write-Error-Message {
	param([string]$Message)
	Write-Host "[FAIL] $Message" -ForegroundColor Red
	$script:BuildFailed = $true
}

function Write-Warning-Message {
	param([string]$Message)
	Write-Host "[WARN] $Message" -ForegroundColor Yellow
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

# The Qt GUI now builds with the msvc2022_64 Qt kit (see below), which needs
# cl.exe/nmake.exe on PATH the same way msbuild does above - same Developer
# Command Prompt/PowerShell precondition, just extended to cover the Qt
# build too. Checked here (not just deep inside the Qt-build block) so a
# missing dev-shell environment fails fast with a clear message instead of a
# confusing qmake/jom error later. Only warns (not fatal) since -SkipGui
# doesn't need this at all.
if (-not $SkipGui) {
	try {
		Get-Command cl -ErrorAction Stop | Out-Null
		Get-Command nmake -ErrorAction Stop | Out-Null
		Write-Success "MSVC compiler tools found (cl/nmake)"
	} catch {
		Write-Warning-Message "cl.exe/nmake.exe not found - Qt GUI build will fail. Run this from a Visual Studio Developer Command Prompt or Developer PowerShell (same requirement as msbuild above)."
	}
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

# A running GUI holds a write lock on RayTracerGUI.exe, and the Qt link step
# below writes straight into RayTracer_Package. Leaving it running produces a
# bare "ld returned 1 exit status" with no hint as to why, so stop it first.
$runningGui = Get-Process -Name 'RayTracerGUI' -ErrorAction SilentlyContinue
if ($runningGui) {
	Write-Host "Stopping running RayTracerGUI.exe (it locks the build output)..."
	Stop-Process -Name 'RayTracerGUI' -Force
	Start-Sleep -Milliseconds 800
}

# Build C++ solution
Write-Header "Building C++ Solution ($Configuration|x64)"
Write-Host "Building: launcher, cpu_renderer, optix_renderer$(if (-not $SkipGui) {', scene_metadata, realtime_renderer'})$(if (-not $SkipTests) {', tests'})"

if ($SkipTests) {
	# Build only the main projects
	msbuild launcher/launcher.vcxproj /p:Configuration=$Configuration /p:Platform=x64 /v:minimal /m
	# scene_metadata.dll/realtime_renderer.dll are NOT launcher.vcxproj
	# ProjectReferences - the Qt GUI loads them at runtime via LoadLibrary
	# instead (see qt_gui/cross_abi_library.h) - so the launcher-only build
	# above never produces them. Without this, the script's own default
	# (-SkipTests, building only launcher) silently produced a GUI whose
	# scene-metadata lookups and Live Preview both no-op at runtime with no
	# build-time error, since the DLLs it LoadLibrary's for simply didn't
	# exist. $SkipTests only exists to skip the (expensive) tests project,
	# not these two - build them here whenever the GUI itself is being
	# built, regardless of $SkipTests.
	if ($LASTEXITCODE -eq 0 -and -not $SkipGui) {
		msbuild scene_metadata/scene_metadata.vcxproj /p:Configuration=$Configuration /p:Platform=x64 /v:minimal /m
		if ($LASTEXITCODE -eq 0) {
			msbuild realtime_renderer/realtime_renderer.vcxproj /p:Configuration=$Configuration /p:Platform=x64 /v:minimal /m
		}
	}
} else {
	# Build entire solution including tests (already covers scene_metadata/realtime_renderer)
	msbuild ray_tracer.sln /p:Configuration=$Configuration /p:Platform=x64 /v:minimal /m
}

if ($LASTEXITCODE -ne 0) {
	Write-Error-Message "C++ build failed"
	exit 1
}

Write-Success "C++ build completed"

# Verify outputs
Write-Header "Verifying Build Outputs"

# Solution-relative paths (x64\$Configuration\...), matching how MSBuild
# actually places outputs when building through ray_tracer.sln (both the
# $SkipTests branch's single-project `msbuild launcher/launcher.vcxproj`
# and the full-solution branch above resolve OutDir the same way once
# $(SolutionDir) is in scope) - NOT project-relative (launcher\x64\...,
# cpu_renderer\x64\...), which these two checks used to assume and which
# doesn't match any actual build output location.
$exePath = "x64\$Configuration\ray_tracer.exe"
if (Test-Path $exePath) {
	$size = (Get-Item $exePath).Length / 1MB
	Write-Success "Launcher: $exePath ($([math]::Round($size, 2)) MB)"
} else {
	# Write-Error-Message (not Write-Error) - this script collects failures
	# via $script:BuildFailed and reports them in one summary at the end;
	# the real Write-Error cmdlet is a terminating error under this script's
	# own $ErrorActionPreference = "Stop", which previously aborted the
	# whole script here (silently skipping the Qt GUI build and deploy
	# below) instead of just flagging this one check and continuing.
	Write-Error-Message "Launcher executable not found: $exePath"
}

$cpuLib = "x64\$Configuration\cpu_renderer.lib"
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
	# bin\$Configuration\, not tests\x64\$Configuration\ - the tests
	# project's OutDir differs from the other three (see the comment above
	# $exePath) and actually lands here.
	$testsExe = "bin\$Configuration\ray_tracer_tests.exe"
	if (Test-Path $testsExe) {
		Write-Success "Tests: $testsExe"
	} else {
		Write-Warning-Message "Tests executable not found: $testsExe"
	}
}

# Build Qt GUI
if (-not $SkipGui) {
	Write-Header "Building Qt GUI"

	# Find Qt installation (msvc2022_64 kit - see BUILD.md for why the GUI
	# moved off MinGW)
	$qtPaths = @(
		"C:\Qt\6.11.1\msvc2022_64\bin",
		"C:\Qt\6.10.0\msvc2022_64\bin",
		"C:\Qt\6.9.0\msvc2022_64\bin"
	)

	$qtBinPath = $null
	foreach ($path in $qtPaths) {
		if (Test-Path "$path\qmake.exe") {
			$qtBinPath = $path
			break
		}
	}

	# Prefer the curated msvc2022_64-first search above over a blind PATH
	# lookup - Get-Command would accept ANY qmake on PATH regardless of
	# which Qt kit it belongs to, including a stale MinGW one left over
	# from before this project switched off MinGW (see BUILD.md). Only
	# fall back to PATH if none of the known kit locations exist, so a
	# MinGW qmake earlier on PATH can never silently override the correct
	# msvc2022_64 one when both are present.
	if ($qtBinPath) {
		$qmake = Get-Command "$qtBinPath\qmake.exe"
	} else {
		$qmake = Get-Command qmake -ErrorAction SilentlyContinue
	}
	if (-not $qmake) {
		Write-Warning-Message "Qt (msvc2022_64 kit) not found in PATH or common locations. Skipping Qt GUI build."
		Write-Host "To build Qt GUI, install the MSVC 2022 64-bit component via the Qt Maintenance Tool, or ensure its bin dir is in PATH."
	} else {
		# jom parallelizes nmake the way mingw32-make -j already did; plain
		# nmake is single-threaded and noticeably slower on a full rebuild,
		# so prefer it when available and fall back to nmake (with a
		# warning) otherwise. Ships with Qt Creator, not the Qt kit itself.
		$jomPath = "C:\Qt\Tools\QtCreator\bin\jom\jom.exe"
		$useJom = Test-Path $jomPath
		if (-not $useJom) {
			Write-Warning-Message "jom.exe not found at $jomPath - falling back to single-threaded nmake."
		}

		Write-Success "Qt: $($qmake.Source)"
		if ($useJom) { Write-Success "jom: $jomPath" }

		Push-Location qt_gui
		try {
			# Add Qt to PATH for this session. cl.exe/nmake.exe (and jom,
			# once found above) are expected to already be on PATH from
			# the Developer Command Prompt/PowerShell this script
			# requires - see the cl/nmake check near the top.
			$env:PATH = "$qtBinPath;$env:PATH"

			# Clean old build. Also removes any stale Makefile/.qmake.stash
			# left over from a prior MinGW build (regenerating against the
			# wrong compiler would silently break in confusing ways), not
			# just on -Clean.
			if (Test-Path .qmake.stash) { Remove-Item .qmake.stash -Force }
			if (Test-Path build) { Remove-Item build -Recurse -Force }
			if (Test-Path Makefile) {
				if ($useJom) { & $jomPath clean 2>$null } else { & nmake clean 2>$null }
			}

			# Generate makefiles
			& $qmake.Source "RayTracerGUI.pro" -spec win32-msvc "CONFIG+=$($Configuration.ToLower())"
			if ($LASTEXITCODE -ne 0) {
				Write-Error-Message "qmake failed"
			} else {
				# Build. jom's /J mirrors mingw32-make -j; all cores rather
				# than a fixed count since this machine may have more, and
				# a smaller one should not be oversubscribed.
				if ($useJom) {
					& $jomPath /J "$([System.Environment]::ProcessorCount)" /F "Makefile.$Configuration"
				} else {
					& nmake /F "Makefile.$Configuration"
				}
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
		Write-Host "  - Run tests:       .\bin\$Configuration\ray_tracer_tests.exe"
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
