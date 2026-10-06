<#
.SYNOPSIS
	Render every gallery scene (the ones other suites skip for "requires external assets") with each scene in its OWN process.
.DESCRIPTION
	GalleryGpuRenderTest (tests/integration/gallery_scenes_render_test.cpp) renders those scenes on the recursive GPU backend and, with -Cpu, on
	the CPU too. Run in a single process with the Large Scenes included, the CPU and GPU copies of the multi-million-triangle scenes pile up
	(16 GB of private memory after six of them) and renders start failing with "out of memory". This script asks the test which scenes would run
	(RT_GALLERY_LIST=1), then runs each one in a fresh process, so memory is returned between scenes, and prints one line per scene.

	All scenes with the CPU render takes about 15 minutes; without -Cpu about 4 minutes (the CPU load of a big scene dominates).
.PARAMETER Cpu
	Also render each scene on the CPU and compare brightness with the GPU (RT_GALLERY_CPU=1).
.PARAMETER SkipLarge
	Leave out the 12 Large Scenes (Power Plant, San Miguel, Rungholt, ...), as the default test run does.
.PARAMETER Filter
	Only scenes whose test name matches this regular expression, e.g. "Sibenik|Zero_Day".
.PARAMETER Configuration
	Release (default) or Debug test executable.
.EXAMPLE
	.\run_gallery_isolated.ps1 -Cpu
#>
param(
	[switch]$Cpu,
	[switch]$SkipLarge,
	[string]$Filter = "",
	[string]$Configuration = "Release"
)

# Not "Stop": under Windows PowerShell 5.1 a native program's stderr text (the renderer logs warnings there) becomes a terminating error when redirected with 2>&1.
# Exit codes are checked explicitly below instead.
$ErrorActionPreference = "Continue"
$exe = Join-Path $PSScriptRoot "..\bin\$Configuration\ray_tracer_tests.exe"
if (-not (Test-Path $exe)) { Write-Host "[FAIL] $exe not found - build the solution first." -ForegroundColor Red; exit 2 }
$exe = (Resolve-Path $exe).Path
Set-Location (Join-Path $PSScriptRoot "..")

if (-not $SkipLarge) { $env:RT_GALLERY_ALL = "1" } else { Remove-Item Env:RT_GALLERY_ALL -ErrorAction SilentlyContinue }
if ($Cpu) { $env:RT_GALLERY_CPU = "1" } else { Remove-Item Env:RT_GALLERY_CPU -ErrorAction SilentlyContinue }

$env:RT_GALLERY_LIST = "1"
$listing = & $exe --gtest_filter="AllScenes/GalleryGpuRenderTest.*" 2>&1 | Out-String
Remove-Item Env:RT_GALLERY_LIST
$names = [regex]::Matches($listing, '(?m)^\[gallery-candidate\] (\S+)') | ForEach-Object { $_.Groups[1].Value }
if ($Filter) { $names = $names | Where-Object { $_ -match $Filter } }
if (-not $names) { Write-Host "No gallery scenes to render (are the scene assets present?)." -ForegroundColor Yellow; exit 0 }
Write-Host "Rendering $(@($names).Count) gallery scene(s), one process each$(if ($Cpu) { ' (CPU + GPU)' } else { ' (GPU)' })..."

$failed = @()
$sw = [Diagnostics.Stopwatch]::StartNew()
foreach ($name in $names) {
	$out = & $exe --gtest_filter="AllScenes/GalleryGpuRenderTest.$name" 2>&1 | Out-String
	$code = $LASTEXITCODE
	$line = [regex]::Match($out, '(?m)^\[gallery\] (.*)$').Groups[1].Value
	if ($code -eq 0) {
		Write-Host "[OK]   $line" -ForegroundColor Green
	} else {
		$why = [regex]::Match($out, '(?m)^(?:\S+\.cpp\(\d+\): error: )?(.*(?:black|darker|brighter|no readable image|render failed|Out of memory).*)$').Groups[1].Value
		Write-Host "[FAIL] $name  $line  $why" -ForegroundColor Red
		$failed += $name
	}
}
Write-Host ""
Write-Host "$(@($names).Count - $failed.Count) of $(@($names).Count) scenes passed in $([int]$sw.Elapsed.TotalSeconds) s" -ForegroundColor Cyan
if ($failed.Count) { Write-Host "Failed: $($failed -join ', ')" -ForegroundColor Red; exit 1 }
exit 0
