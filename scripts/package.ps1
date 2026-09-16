<#
.SYNOPSIS
    Build and package a distributable Ray Tracer release, in one of three tiers.
.DESCRIPTION
    Tier   | Contents                                          | Target machine needs
    -------|---------------------------------------------------|---------------------------------
    Lite   | ray_tracer.exe (CLI) only, CPU rendering           | Nothing beyond VC++ redist
    Medium | + RayTracerGUI.exe, CPU rendering only             | + Qt runtime (bundled)
    Full   | + GPU rendering (Live Preview, OptiX/CUDA backend) | + an NVIDIA GPU/driver

    Tiers are additive (Medium contains everything Lite does, Full contains
    everything Medium does) and are assembled into separate output folders
    (RayTracer_Package_Lite/_Medium/_Full) rather than overwriting each
    other or the shared dev-build RayTracer_Package/, so more than one can
    exist side by side.

    The CUDA runtime is statically linked into every binary that needs it
    (cudart_static.lib - see launcher.vcxproj's and realtime_renderer.vcxproj's
    own AdditionalDependencies) - no cudart64_*.dll is ever required on the
    target machine, only the NVIDIA driver's own nvcuda.dll (never bundled;
    assumed present with any working GPU driver install, which is also the
    prerequisite for OptiX itself - it has no redistributable driver-
    independent runtime and JIT-compiles this package's own bundled .ptx
    files against whatever driver is installed at launch time). Full's GPU
    dependency is therefore just "a working NVIDIA driver", not a CUDA
    Toolkit install - an earlier version of this script searched for and
    bundled cudart64_*.dll, which no binary here actually imports.

    Medium deliberately cannot exclude Qt6Multimedia.dll/its ffmpeg backend:
    RayTracerGUI.pro declares `QT += multimedia multimediawidgets` (for the
    Preview tab's embedded video playback), which is a hard, compile-time
    DLL import baked into RayTracerGUI.exe itself - stripping those DLLs
    post-build would make the exe fail to even load, not just disable one
    feature. Medium vs Full is a CPU-vs-GPU rendering split, not a
    video-vs-no-video one; there is no packaging-only way to build a smaller
    GUI tier without in-app video preview, only a source-level one (a qmake
    CONFIG flag conditionally compiling the feature out).
.PARAMETER Tier
    Lite, Medium, or Full (default Full).
.PARAMETER Configuration
    Debug or Release (default Release).
.PARAMETER SkipBuild
    Skip the build step and just (re)assemble the package from whatever is
    already built in x64\$Configuration - useful for iterating on this
    script itself without a full rebuild each time.
.PARAMETER Zip
    Also produce releases\RayTracer_<Tier>_<yyyyMMdd>.zip from the
    assembled folder.
.EXAMPLE
    .\scripts\package.ps1 -Tier Lite
    .\scripts\package.ps1 -Tier Full -Zip
#>

param(
	[ValidateSet("Lite", "Medium", "Full")]
	[string]$Tier = "Full",
	[ValidateSet("Debug", "Release")]
	[string]$Configuration = "Release",
	[switch]$SkipBuild,
	[switch]$Zip
)

$ErrorActionPreference = "Stop"

# $PSScriptRoot is scripts/, so the repo root is one level up - see every
# other script in this folder for the same convention.
$RepoRoot = Split-Path $PSScriptRoot -Parent
$BuildDir = Join-Path $RepoRoot "x64\$Configuration"
$OutputDir = Join-Path $RepoRoot "RayTracer_Package_$Tier"

# MSVC-first Qt kit search, shared by the build step and the windeployqt
# step below - see deploy_qt_gui.ps1's own comment for why this ordered list
# (not a bare PATH lookup) matters: a stale MinGW qmake/windeployqt earlier
# on PATH would silently produce an ABI-mismatched, crash-at-launch build,
# passing every naive check along the way.
$qtCandidates = @(
	"C:\Qt\6.11.1\msvc2022_64",
	"C:\Qt\6.10.0\msvc2022_64",
	"C:\Qt\6.9.0\msvc2022_64"
)
$qtRoot = $null
foreach ($c in $qtCandidates) {
	if (Test-Path "$c\bin\qmake.exe") { $qtRoot = $c; break }
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Ray Tracer Packaging - Tier: $Tier ($Configuration)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

function Assert-LastExitCode([string]$what) {
	if ($LASTEXITCODE -ne 0) { throw "$what failed (exit code $LASTEXITCODE)" }
}

# ============================================================================
# 1. Build
# ============================================================================
if (-not $SkipBuild) {
	# A running instance locks its own exe/DLLs, breaking the copy step
	# below with a confusing "file in use" error instead of a clear one -
	# same guard build_all.ps1 already uses.
	$runningGui = Get-Process -Name 'RayTracerGUI' -ErrorAction SilentlyContinue
	if ($runningGui) {
		Write-Host "`nStopping running RayTracerGUI.exe (it locks the build output)..."
		Stop-Process -Name 'RayTracerGUI' -Force
		Start-Sleep -Milliseconds 500
	}

	Write-Host "`n[Build] launcher (CLI)..." -ForegroundColor Cyan
	# /p:SolutionDir explicit on every individual-project msbuild call below -
	# MSBuild only auto-populates $(SolutionDir) when building THROUGH a
	# .sln; invoking a single .vcxproj directly (as every call here does)
	# leaves it resolving to that project's own directory instead (e.g.
	# "launcher\"), which silently sends its post-build "copy to
	# RayTracer_Package" step to a wrong nested "launcher\RayTracer_Package\"
	# that the real package never sees - confirmed to actually happen, not
	# just a theoretical risk. This global property also propagates to
	# whatever the project references transitively (optix_renderer.vcxproj,
	# built as a dependency of launcher.vcxproj), fixing its own PTX
	# post-build copy the same way.
	& msbuild "$RepoRoot\launcher\launcher.vcxproj" /p:Configuration=$Configuration /p:Platform=x64 /p:SolutionDir="$RepoRoot\" /v:minimal /m
	Assert-LastExitCode "launcher build"

	if ($Tier -ne "Lite") {
		Write-Host "`n[Build] scene_metadata (scene registry, needed by the GUI regardless of CPU/GPU mode)..." -ForegroundColor Cyan
		& msbuild "$RepoRoot\scene_metadata\scene_metadata.vcxproj" /p:Configuration=$Configuration /p:Platform=x64 /p:SolutionDir="$RepoRoot\" /v:minimal /m
		Assert-LastExitCode "scene_metadata build"

		if ($Tier -eq "Full") {
			Write-Host "`n[Build] realtime_renderer (GPU Live Preview)..." -ForegroundColor Cyan
			& msbuild "$RepoRoot\realtime_renderer\realtime_renderer.vcxproj" /p:Configuration=$Configuration /p:Platform=x64 /p:SolutionDir="$RepoRoot\" /v:minimal /m
			Assert-LastExitCode "realtime_renderer build"
		}

		Write-Host "`n[Build] Qt GUI..." -ForegroundColor Cyan
		if (-not $qtRoot) {
			throw "Qt (msvc2022_64 kit) not found in any of: $($qtCandidates -join ', '). Install it via the Qt Maintenance Tool, or edit `$qtCandidates at the top of this script."
		}
		$env:PATH = "$qtRoot\bin;$env:PATH"
		Push-Location "$RepoRoot\qt_gui"
		try {
			if (Test-Path .qmake.stash) { Remove-Item .qmake.stash -Force }
			& "$qtRoot\bin\qmake.exe" "RayTracerGUI.pro" -spec win32-msvc "CONFIG+=$($Configuration.ToLower())"
			Assert-LastExitCode "qmake"
			& nmake /F "Makefile.$Configuration"
			Assert-LastExitCode "Qt GUI build (nmake)"
		} finally {
			Pop-Location
		}
	}
	Write-Host "`n[Build] Done.`n" -ForegroundColor Green
} else {
	Write-Host "`n[Build] Skipped (-SkipBuild) - packaging whatever is already in $BuildDir`n" -ForegroundColor Yellow
}

# ============================================================================
# 2. Assemble the package
# ============================================================================
Write-Host "[Package] Assembling $Tier package -> $OutputDir" -ForegroundColor Cyan
if (Test-Path $OutputDir) { Remove-Item -Recurse -Force $OutputDir }
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
New-Item -ItemType Directory -Path "$OutputDir\output" -Force | Out-Null

function Copy-RequiredFile([string]$Src, [string]$DstName) {
	if (-not (Test-Path $Src)) {
		throw "Required file missing: $Src`n(did the build step run, or did -SkipBuild skip a build this tier actually needs?)"
	}
	if (-not $DstName) { $DstName = Split-Path $Src -Leaf }
	Copy-Item $Src (Join-Path $OutputDir $DstName) -Force
	Write-Host "  OK $DstName" -ForegroundColor Green
}

# Every tier: the CLI itself, renamed to match this project's own existing
# "RayTracer.exe" distribution-name convention (releases/README.md's own
# package-structure example).
Copy-RequiredFile "$BuildDir\ray_tracer.exe" "RayTracer.exe"

if ($Tier -ne "Lite") {
	Copy-RequiredFile "$BuildDir\RayTracerGUI.exe"
	Copy-RequiredFile "$BuildDir\scene_metadata.dll"

	if ($Tier -eq "Full") {
		Copy-RequiredFile "$BuildDir\realtime_renderer.dll"
		Copy-RequiredFile "$RepoRoot\gpu\optix\optix_programs.ptx"
		# wavefront_programs.ptx only exists as a separate file on some
		# branches/configurations of this project - optional, not required,
		# unlike optix_programs.ptx above.
		if (Test-Path "$RepoRoot\gpu\optix\wavefront_programs.ptx") {
			Copy-Item "$RepoRoot\gpu\optix\wavefront_programs.ptx" $OutputDir -Force
			Write-Host "  OK wavefront_programs.ptx" -ForegroundColor Green
		}
	}

	Write-Host "`n[Package] Deploying Qt runtime via windeployqt..." -ForegroundColor Cyan
	if (-not $qtRoot) {
		throw "Qt (msvc2022_64 kit) not found in any of: $($qtCandidates -join ', ') - cannot deploy Qt runtime for the $Tier tier."
	}
	$windeployqtPath = "$qtRoot\bin\windeployqt.exe"
	Push-Location $OutputDir
	try {
		# --force: see deploy_qt_gui.ps1's own comment - without it,
		# windeployqt skips overwriting a Qt6*.dll that already exists at
		# the destination regardless of whether it's the right build, which
		# let a stale MinGW-built DLL survive a "successful" deploy before.
		& $windeployqtPath "RayTracerGUI.exe" --no-translations --no-compiler-runtime --force | Out-Null
		Assert-LastExitCode "windeployqt"
	} finally {
		Pop-Location
	}
	Write-Host "  OK Qt runtime (platforms/styles/imageformats/multimedia plugins + Qt6*.dll)" -ForegroundColor Green
}

# README + launcher batch - templates live in this repo (scripts/templates/)
# rather than being generated inline here, so their wording can be edited
# without touching this script.
$templateDir = Join-Path $PSScriptRoot "templates"
$readmeSrc = Join-Path $templateDir "README_PACKAGE.txt"
if (Test-Path $readmeSrc) {
	(Get-Content $readmeSrc -Raw) -replace '\{TIER\}', $Tier | Set-Content (Join-Path $OutputDir "README.txt") -NoNewline
	Write-Host "  OK README.txt" -ForegroundColor Green
} else {
	Write-Host "  [WARNING] $readmeSrc not found - package will ship without a README.txt" -ForegroundColor Yellow
}
$launcherSrc = Join-Path $templateDir "launcher.bat"
if (Test-Path $launcherSrc) {
	Copy-Item $launcherSrc (Join-Path $OutputDir "launcher.bat") -Force
	Write-Host "  OK launcher.bat" -ForegroundColor Green
} else {
	Write-Host "  [WARNING] $launcherSrc not found - package will ship without launcher.bat" -ForegroundColor Yellow
}

# Visual C++ runtime - the one genuinely external, non-statically-linkable
# dependency every tier has (the CLI and, for Medium/Full, the GUI are both
# built with the dynamic CRT). Best-effort: a target machine almost always
# already has this from Windows Update or another app's installer, but
# bundling it removes that as a possible "won't launch" support question.
Write-Host "`n[Package] Searching for Visual C++ runtime DLLs..." -ForegroundColor Cyan
$vcRedistPaths = @(
	"C:\Program Files\Microsoft Visual Studio\2026\Community\VC\Redist\MSVC\14.50.34831\x64\Microsoft.VC145.CRT",
	"C:\Windows\System32"
)
foreach ($vcDll in @("vcruntime140_1.dll", "vcruntime140.dll", "msvcp140.dll")) {
	$found = $false
	foreach ($path in $vcRedistPaths) {
		$dllPath = Join-Path $path $vcDll
		if (Test-Path $dllPath) {
			Copy-Item $dllPath $OutputDir -Force -ErrorAction SilentlyContinue
			Write-Host "  OK $vcDll" -ForegroundColor Green
			$found = $true
			break
		}
	}
	if (-not $found) {
		$systemDll = (Get-Command $vcDll -ErrorAction SilentlyContinue).Path
		if ($systemDll) {
			Copy-Item $systemDll $OutputDir -Force
			Write-Host "  OK $vcDll (from system PATH)" -ForegroundColor Green
		} else {
			Write-Host "  [WARNING] $vcDll not found - target machine will need the VC++ redistributable installed" -ForegroundColor Yellow
		}
	}
}

$sizeMB = [math]::Round(((Get-ChildItem $OutputDir -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB), 1)
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Package assembled: $OutputDir ($sizeMB MB)" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan

# ============================================================================
# 3. Zip (optional)
# ============================================================================
if ($Zip) {
	$releasesDir = Join-Path $RepoRoot "releases"
	if (-not (Test-Path $releasesDir)) { New-Item -ItemType Directory -Path $releasesDir -Force | Out-Null }
	$zipPath = Join-Path $releasesDir "RayTracer_${Tier}_$(Get-Date -Format yyyyMMdd).zip"
	if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
	Write-Host "`n[Zip] Compressing to $zipPath ..." -ForegroundColor Cyan
	Compress-Archive -Path "$OutputDir\*" -DestinationPath $zipPath
	$zipMB = [math]::Round(((Get-Item $zipPath).Length / 1MB), 1)
	Write-Host "  OK $zipPath ($zipMB MB)" -ForegroundColor Green
}

Write-Host "`nTo test the package:"
Write-Host "  cd `"$OutputDir`""
if ($Tier -eq "Lite") {
	Write-Host "  .\RayTracer.exe --help"
} else {
	Write-Host "  .\launcher.bat"
}
if (-not $Zip) {
	Write-Host "`nTo create a ZIP for distribution, re-run with -Zip, or:"
	Write-Host "  Compress-Archive -Path `"$OutputDir\*`" -DestinationPath RayTracer_${Tier}.zip"
}
