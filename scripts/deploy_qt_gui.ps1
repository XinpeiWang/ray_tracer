# Deploy Ray Tracer Qt GUI with all dependencies
# This script focuses on Qt GUI and its dependencies.
# Backend (ray_tracer.exe) and PTX are now auto-deployed by MSBuild post-build events.

param(
	[ValidateSet("Debug", "Release")]
	[string]$Configuration = "Release"
)

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Ray Tracer Qt GUI Deployment Script" -ForegroundColor Cyan
Write-Host "Configuration: $Configuration" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# $PSScriptRoot is scripts/, so the repo root is one level up. Getting this
# wrong is quiet rather than loud: paths still resolve, just to a directory
# that does not exist, and the script reports missing files instead of
# failing outright.
$repoRoot = Split-Path $PSScriptRoot -Parent
$packageDir = "$repoRoot\RayTracer_Package"

# Verify backend files exist (they should be auto-deployed by MSBuild)
$backendExe = "$packageDir\ray_tracer.exe"
$ptxFile = "$packageDir\optix_programs.ptx"

if (-not (Test-Path $backendExe)) {
	Write-Host "[WARNING] Backend launcher not found at: $backendExe" -ForegroundColor Yellow
	Write-Host "          The launcher post-build event should auto-deploy it." -ForegroundColor Yellow
	Write-Host "          Make sure you build the launcher project." -ForegroundColor Yellow
}

if (-not (Test-Path $ptxFile)) {
	Write-Host "[WARNING] PTX shader not found at: $ptxFile" -ForegroundColor Yellow
	Write-Host "          The optix_renderer post-build event should auto-deploy it." -ForegroundColor Yellow
	Write-Host "          Make sure you build the optix_renderer project." -ForegroundColor Yellow
}

if (-not (Test-Path $packageDir)) {
	New-Item -ItemType Directory -Path $packageDir | Out-Null
	Write-Host "[INFO] Created package directory: $packageDir`n" -ForegroundColor Green
}

Write-Host "[Step 1/2] Copying Qt GUI executable..." -ForegroundColor Cyan
$qtGuiBuildDir = "$repoRoot\qt_gui\$($Configuration.ToLower())"
$qtGuiSourceExe = "$qtGuiBuildDir\RayTracerGUI.exe"

# Qt build outputs to ../RayTracer_Package directly
if (-not (Test-Path $qtGuiSourceExe)) {
	# Check if it was built directly into the package
	$qtGuiExe = "$packageDir\RayTracerGUI.exe"
	if (Test-Path $qtGuiExe) {
		Write-Host "      => RayTracerGUI.exe already in package`n" -ForegroundColor Gray
	} else {
		Write-Host "      [WARNING] Qt GUI executable not found!" -ForegroundColor Yellow
		Write-Host "      Expected: $qtGuiSourceExe" -ForegroundColor Yellow
		Write-Host "      You need to build the Qt GUI first using qmake/Qt Creator.`n" -ForegroundColor Yellow
	}
} else {
	# Copy from build directory
	Copy-Item $qtGuiSourceExe "$packageDir\RayTracerGUI.exe" -Force
	$guiInfo = Get-Item "$packageDir\RayTracerGUI.exe"
	Write-Host "      => RayTracerGUI.exe ($([math]::Round($guiInfo.Length/1KB)) KB)`n" -ForegroundColor Gray
}

Write-Host "[Step 2/2] Deploying Qt dependencies with windeployqt..." -ForegroundColor Cyan

$qtGuiExe = "$packageDir\RayTracerGUI.exe"
if (-not (Test-Path $qtGuiExe)) {
	Write-Host "      [ERROR] Cannot deploy Qt dependencies - GUI executable not found" -ForegroundColor Red
	exit 1
}

# Find windeployqt (try multiple Qt versions/locations)
$windeployqtPaths = @(
	"C:\Qt\6.11.1\msvc2022_64\bin\windeployqt.exe",
	"C:\Qt\6.10.0\msvc2022_64\bin\windeployqt.exe",
	"C:\Qt\6.9.0\msvc2022_64\bin\windeployqt.exe",
	"C:\Qt\6.11.1\mingw_64\bin\windeployqt.exe",
	"C:\Qt\6.10.0\mingw_64\bin\windeployqt.exe",
	"C:\Qt\6.9.0\mingw_64\bin\windeployqt.exe"
)

# Check the explicit, MSVC-first ordered list BEFORE a blind PATH lookup.
# Get-Command would accept ANY windeployqt on PATH regardless of which Qt
# kit it belongs to, including a stale MinGW one left over from before this
# project switched off MinGW (see BUILD.md) - packaging MinGW-flavored Qt
# DLLs against an MSVC-built RayTracerGUI.exe is an ABI mismatch that
# crashes at launch, with this script itself reporting success. Only fall
# back to a bare PATH lookup if none of the known kit locations exist.
$windeployqtPath = $null
foreach ($path in $windeployqtPaths) {
	if (Test-Path $path) {
		$windeployqtPath = $path
		break
	}
}
if (-not $windeployqtPath) {
	$windeployqt = Get-Command windeployqt -ErrorAction SilentlyContinue
	if ($windeployqt) { $windeployqtPath = $windeployqt.Source }
}

if (-not $windeployqtPath) {
	Write-Host "      [WARNING] windeployqt not found. Qt DLLs will be missing!" -ForegroundColor Yellow
	Write-Host "      To manually deploy Qt dependencies, run:" -ForegroundColor Yellow
	Write-Host "      cd $packageDir" -ForegroundColor Yellow
	Write-Host "      <qt-install-path>\bin\windeployqt.exe RayTracerGUI.exe --no-translations`n" -ForegroundColor Yellow
} else {
	Write-Host "      Using: $windeployqtPath" -ForegroundColor Gray

	Push-Location $packageDir
	try {
		# --force: windeployqt otherwise skips overwriting a Qt6*.dll that
		# already exists at the destination, regardless of whether it's the
		# right build - a leftover MinGW-built Qt6Core.dll/Qt6Multimedia.dll/
		# etc. from before this project's MSVC switch survived silently
		# through a "successful" deploy this way (confirmed live: the app
		# ran with no window and exited with STATUS_DLL_NOT_FOUND once the
		# MinGW runtime DLLs it actually needed were removed as dead
		# weight). Without --force, switching Qt toolchains ever again
		# would reintroduce this exact bug on any machine with a package
		# already deployed from the old toolchain.
		# No `2>&1` here: this script sets $ErrorActionPreference = "Stop",
		# and merging a native command's stderr into the success stream
		# turns EVERY stderr line into a terminating PowerShell error under
		# that preference - windeployqt routinely writes harmless warnings
		# to stderr (e.g. "Cannot find dxcompiler.dll... not a problem
		# unless Direct3D 12 ... is used"), and with the merge in place any
		# one of them silently aborted this whole script before it ever
		# reached the DLL verification below. Letting stderr print directly
		# to the console (unredirected) is fine - we only branch on
		# $LASTEXITCODE afterward, never on captured output content.
		& $windeployqtPath "RayTracerGUI.exe" --no-translations --no-compiler-runtime --force | Out-Null
		if ($LASTEXITCODE -eq 0) {
			Write-Host "      => Qt dependencies deployed successfully`n" -ForegroundColor Green

			# Verify critical DLLs exist AND are actually MSVC-built, not
			# just present - a stale MinGW-built file left in place (see
			# the --force comment above) passes a plain Test-Path check
			# while still being completely broken at runtime. Grepping for
			# the MinGW runtime import names is a crude but effective
			# proxy for "wrong toolchain" without needing dumpbin/a VS
			# developer environment to be available here.
			$criticalDlls = @("Qt6Core.dll", "Qt6Gui.dll", "Qt6Widgets.dll", "Qt6Multimedia.dll", "Qt6MultimediaWidgets.dll", "Qt6Network.dll", "Qt6Svg.dll")
			$missingDlls = @()
			$mingwDlls = @()
			foreach ($dll in $criticalDlls) {
				if (-not (Test-Path $dll)) {
					$missingDlls += $dll
				} elseif (Select-String -Path $dll -Pattern "libgcc_s_seh|libwinpthread|libstdc\+\+" -Quiet -ErrorAction SilentlyContinue) {
					$mingwDlls += $dll
				}
			}

			if ($missingDlls.Count -gt 0) {
				Write-Host "      [WARNING] Some Qt DLLs are still missing:" -ForegroundColor Yellow
				foreach ($dll in $missingDlls) {
					Write-Host "        - $dll" -ForegroundColor Yellow
				}
			}
			if ($mingwDlls.Count -gt 0) {
				Write-Host "      [WARNING] Some deployed Qt DLLs are still MinGW-built (will crash on launch):" -ForegroundColor Red
				foreach ($dll in $mingwDlls) {
					Write-Host "        - $dll" -ForegroundColor Red
				}
			}
		} else {
			Write-Host "      [WARNING] windeployqt failed (exit code: $LASTEXITCODE)" -ForegroundColor Yellow
		}
	} finally {
		Pop-Location
	}
}

Write-Host "========================================" -ForegroundColor Green
Write-Host "Deployment Complete!" -ForegroundColor Green
Write-Host "========================================`n" -ForegroundColor Green

# Final validation
$finalChecks = @(
	@{Name="Launcher"; Path="$packageDir\ray_tracer.exe"},
	@{Name="PTX Shader"; Path="$packageDir\optix_programs.ptx"},
	@{Name="Qt GUI"; Path="$packageDir\RayTracerGUI.exe"},
	@{Name="Qt6Core.dll"; Path="$packageDir\Qt6Core.dll"}
)

$allGood = $true
foreach ($check in $finalChecks) {
	$exists = Test-Path $check.Path
	$symbol = if ($exists) { "✓" } else { "✗"; $allGood = $false }
	$color = if ($exists) { "Green" } else { "Red" }
	Write-Host "  $symbol $($check.Name)" -ForegroundColor $color
}

Write-Host ""
if ($allGood) {
	Write-Host "All files present! Package is ready to run.`n" -ForegroundColor Green
	Write-Host "To launch the GUI:" -ForegroundColor Cyan
	Write-Host "  Start-Process `"$packageDir\RayTracerGUI.exe`"`n" -ForegroundColor White
} else {
	Write-Host "Some files are missing. Please review the warnings above.`n" -ForegroundColor Yellow
}

Write-Host "Package location: $packageDir" -ForegroundColor Gray
