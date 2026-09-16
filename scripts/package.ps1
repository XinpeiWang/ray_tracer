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
    everything Medium does) and are all assembled directly into the shared
    dev-build RayTracer_Package\ folder - the same folder RayTracerGUI.pro's
    own DESTDIR and every MSBuild post-build step already write into - rather
    than a separate RayTracer_Package_<Tier>\ copy. This used to create a
    second folder per tier so more than one could exist side by side, but in
    practice only one tier is ever built on a given machine, so the separate
    copy was just duplicated disk space and a source of "which folder do I
    actually run" confusion. $OutputDir itself is never wiped/pruned - a
    previous, higher tier's files can still be sitting there - but -Zip's
    output always matches what the current -Tier promises: it excludes any
    higher-tier files at zip time rather than deleting them from the
    folder (RayTracerGUI.exe and its Qt runtime have no other copy to
    restore from, so physically removing them to "enforce" a smaller tier
    would destroy the only copy of that build). Running the assembled
    folder directly, without -Zip, can still launch more than the
    requested tier if a higher tier was built there previously; the script
    warns about this case.

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
# Same folder RayTracerGUI.pro's DESTDIR and every MSBuild post-build step
# already write into - see the .DESCRIPTION comment above for why this is
# no longer a separate RayTracer_Package_$Tier\ copy.
$OutputDir = Join-Path $RepoRoot "RayTracer_Package"

# MSVC-first Qt kit search, needed to find qmake.exe for the build step
# below (deploy_qt_gui.ps1, called later for the windeployqt step, does its
# own equivalent search for windeployqt.exe) - see deploy_qt_gui.ps1's own
# comment for why this ordered list (not a bare PATH lookup) matters: a
# stale MinGW qmake earlier on PATH would silently produce an
# ABI-mismatched, crash-at-launch build, passing every naive check along
# the way.
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

# A running instance locks its own exe/DLLs - unconditional (not just under
# -SkipBuild) because the windeployqt step below (via deploy_qt_gui.ps1) can
# hit the exact same "file in use" failure even when the build itself is
# skipped.
$runningGui = Get-Process -Name 'RayTracerGUI' -ErrorAction SilentlyContinue
if ($runningGui) {
	Write-Host "`nStopping running RayTracerGUI.exe (it locks the build output)..."
	Stop-Process -Name 'RayTracerGUI' -Force
	Start-Sleep -Milliseconds 500
}

# ============================================================================
# 1. Build
# ============================================================================
# This build phase intentionally parallels build_all.ps1's own msbuild/qmake
# invocations rather than calling it directly: build_all.ps1 has no flag
# combination for "GUI without GPU" (this script's Medium tier), and its
# other flags (-SkipTests, -Clean, ...) don't map cleanly onto tiers. If you
# change how a project here gets built (a new /p: flag, a jom/nmake switch,
# a prerequisite check), check whether the equivalent spot in build_all.ps1
# needs the same change - this is deliberate, known duplication, not an
# oversight.
if (-not $SkipBuild) {
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
			# jom parallelizes nmake - see build_all.ps1's own comment for why
			# (ships with Qt Creator, not the Qt kit itself, hence the
			# separate, fixed install path rather than a PATH lookup).
			$jomPath = "C:\Qt\Tools\QtCreator\bin\jom\jom.exe"
			if (Test-Path $jomPath) {
				& $jomPath /J "$([System.Environment]::ProcessorCount)" /F "Makefile.$Configuration"
				Assert-LastExitCode "Qt GUI build (jom)"
			} else {
				& nmake /F "Makefile.$Configuration"
				Assert-LastExitCode "Qt GUI build (nmake)"
			}
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
# No longer wiped first - $OutputDir IS the live qmake/MSBuild build output
# folder now (see above), and this step runs AFTER the build, so wiping it
# here would delete the very files being packaged. windeployqt's own
# --force flag and every Copy-Item -Force call below already handle
# overwriting stale content from a previous packaging run; only truly
# orphaned files (e.g. a Qt plugin removed between Qt versions) could go
# stale, which isn't worth a destructive wipe to avoid.
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
New-Item -ItemType Directory -Path "$OutputDir\output" -Force | Out-Null

# Tier contract enforcement is handled at zip time (see the Zip section
# below), NOT here by deleting files from $OutputDir. $OutputDir doubles as
# qmake's own DESTDIR - RayTracerGUI.exe and its windeployqt'd Qt runtime
# have no other copy anywhere in the repo (unlike realtime_renderer.dll/
# optix_programs.ptx, which are also copied from $BuildDir/gpu\optix\ and
# so could be safely deleted-and-recopied) - so physically removing them
# here to "enforce Lite" would destroy the only copy of that build, and a
# later `-Tier Full -SkipBuild` could not restore it without a real
# qmake/nmake rebuild. Excluding them from the zip is sufficient to fix the
# actual problem (a shipped Lite/Medium zip must not contain higher-tier
# files) without that risk.

function Copy-RequiredFile([string]$Src, [string]$DstName) {
	if (-not (Test-Path $Src)) {
		throw "Required file missing: $Src`n(did the build step run, or did -SkipBuild skip a build this tier actually needs?)"
	}
	if (-not $DstName) { $DstName = Split-Path $Src -Leaf }
	$dst = Join-Path $OutputDir $DstName
	# RayTracerGUI.exe's own source IS $OutputDir now (see the Tier -ne
	# Lite block below) - Copy-Item errors on a same-file copy, and none is
	# needed since it's already exactly where it belongs. Test-Path guards
	# the Resolve-Path call so this never evaluates .Path against $null
	# (Resolve-Path -ErrorAction SilentlyContinue returns $null, not an
	# error, when $dst doesn't exist yet - e.g. a brand new checkout).
	if ((Test-Path $dst) -and (Resolve-Path $Src).Path -eq (Resolve-Path $dst).Path) {
		Write-Host "  OK $DstName (already in place)" -ForegroundColor Green
		return
	}
	Copy-Item $Src $dst -Force
	Write-Host "  OK $DstName" -ForegroundColor Green
}

# Every tier: the CLI itself, renamed to match this project's own existing
# "RayTracer.exe" distribution-name convention (releases/README.md's own
# package-structure example).
Copy-RequiredFile "$BuildDir\ray_tracer.exe" "RayTracer.exe"

if ($Tier -ne "Lite") {
	# NOT $BuildDir - RayTracerGUI.pro's own DESTDIR sends qmake/nmake's
	# link output straight to RayTracer_Package\ (confirmed by this
	# project's own build log: "link ... /OUT: ..\RayTracer_Package\
	# RayTracerGUI.exe"), never to x64\$Configuration\ the way every
	# MSBuild-based .vcxproj target here does. That's the same folder
	# $OutputDir now points at, so Copy-RequiredFile's same-path check
	# above turns this into a no-op existence check, not a real copy.
	Copy-RequiredFile "$RepoRoot\RayTracer_Package\RayTracerGUI.exe"
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

	Write-Host "`n[Package] Deploying Qt runtime..." -ForegroundColor Cyan
	# Delegate to deploy_qt_gui.ps1 rather than reimplementing windeployqt
	# here - it already does everything this needs (MSVC-first Qt-kit
	# search, --force, and critically a post-deploy check that greps
	# deployed DLLs for MinGW runtime import strings to catch a stale
	# MinGW-built Qt6*.dll surviving --force, which this block used to have
	# no equivalent for). One Qt-deployment code path instead of two also
	# means build_all.ps1 -Deploy and this script can never disagree on
	# which Qt version got deployed into the shared RayTracer_Package\.
	& "$PSScriptRoot\deploy_qt_gui.ps1" -Configuration $Configuration
	Assert-LastExitCode "deploy_qt_gui.ps1"
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

# $OutputDir itself is never wiped/pruned (see above - RayTracerGUI.exe and
# its Qt runtime have no other copy to restore from), so a higher tier's
# files from a previous run can still be sitting here even though $Tier
# asked for less. That's harmless for -Zip (excluded above), but running
# the folder directly (e.g. .\launcher.bat) would still launch/include
# more than the requested tier - flag it instead of leaving it silent.
$leftoverHigherTier = @()
if ($Tier -eq "Lite" -and (Test-Path (Join-Path $OutputDir "RayTracerGUI.exe"))) { $leftoverHigherTier += "RayTracerGUI.exe + Qt runtime (from a previous Medium/Full build)" }
if ($Tier -ne "Full" -and (Test-Path (Join-Path $OutputDir "realtime_renderer.dll"))) { $leftoverHigherTier += "realtime_renderer.dll + GPU shaders (from a previous Full build)" }
if ($leftoverHigherTier.Count -gt 0) {
	Write-Host "`n[WARNING] $OutputDir still contains higher-tier files from a previous run:" -ForegroundColor Yellow
	foreach ($item in $leftoverHigherTier) { Write-Host "  - $item" -ForegroundColor Yellow }
	Write-Host "  These are excluded from -Zip output but still present if you run/ship this folder directly." -ForegroundColor Yellow
}

# ============================================================================
# 3. Zip (optional)
# ============================================================================
if ($Zip) {
	$releasesDir = Join-Path $RepoRoot "releases"
	if (-not (Test-Path $releasesDir)) { New-Item -ItemType Directory -Path $releasesDir -Force | Out-Null }
	$zipPath = Join-Path $releasesDir "RayTracer_${Tier}_$(Get-Date -Format yyyyMMdd).zip"
	if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
	Write-Host "`n[Zip] Compressing to $zipPath ..." -ForegroundColor Cyan
	# $OutputDir doubles as the everyday dev-build/test folder (see the
	# .DESCRIPTION comment at the top of this script), so it can accumulate
	# local render output and debug artifacts that must never ship in a
	# release - same dev-cruft vocabulary .gitignore already uses for this
	# exact folder (output/, test_*, debug_*, verify_*, working_*,
	# cornell_*, *.run_marker.txt) - and, since $OutputDir is never wiped
	# (see above), it can also still hold a HIGHER tier's files from a
	# previous run (e.g. Full built yesterday, Lite built today). Exclude
	# both instead of zipping $OutputDir wholesale, so the zip always
	# matches what -Tier actually promises regardless of build order or
	# what a previous run left behind.
	$zipExcludeNames = [System.Collections.Generic.List[string]]@("output")
	$zipExcludePatterns = @("test_*", "debug_*", "verify_*", "working_*", "cornell_*", "*.run_marker.txt", "*.ppm")
	if ($Tier -eq "Lite") {
		$zipExcludeNames.AddRange([string[]]@("RayTracerGUI.exe", "scene_metadata.dll", "realtime_renderer.dll", "optix_programs.ptx", "wavefront_programs.ptx", "platforms", "styles", "imageformats", "multimedia", "iconengines", "generic", "networkinformation", "tls"))
		# Allow-list, not a Qt6*.dll deny-list: windeployqt also drops non-
		# Qt6-prefixed runtime deps alongside Qt6*.dll (ffmpeg codecs, the
		# ANGLE/D3D compiler, the opengl32sw software rasterizer - exact set
		# varies by Qt version/enabled modules), so enumerating every
		# Qt-side DLL by name would silently miss new ones on a Qt upgrade.
		# Lite's CLI only ever needs the VC++ redist DLLs (statically
		# linked CUDA runtime, per this file's own .DESCRIPTION).
		$liteDllAllowList = @("vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll")
		Get-ChildItem $OutputDir -Filter "*.dll" -ErrorAction SilentlyContinue |
			Where-Object { $liteDllAllowList -notcontains $_.Name } |
			ForEach-Object { $zipExcludeNames.Add($_.Name) }
	} elseif ($Tier -eq "Medium") {
		$zipExcludeNames.AddRange([string[]]@("realtime_renderer.dll", "optix_programs.ptx", "wavefront_programs.ptx"))
	}
	$itemsToZip = Get-ChildItem $OutputDir | Where-Object {
		$name = $_.Name
		if ($zipExcludeNames -contains $name) { return $false }
		foreach ($pattern in $zipExcludePatterns) {
			if ($name -like $pattern) { return $false }
		}
		return $true
	}
	Compress-Archive -Path $itemsToZip.FullName -DestinationPath $zipPath
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
	Write-Host "`nTo create a ZIP for distribution, re-run with -Zip"
	Write-Host "  (not a manual Compress-Archive on $OutputDir directly - that"
	Write-Host "  would also bundle any local render output/test files sitting"
	Write-Host "  in this shared dev-build folder; -Zip excludes those)."
}
