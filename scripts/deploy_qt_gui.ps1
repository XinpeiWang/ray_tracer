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

$repoRoot = $PSScriptRoot
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
	"C:\Qt\6.11.1\mingw_64\bin\windeployqt.exe",
	"C:\Qt\6.10.0\mingw_64\bin\windeployqt.exe",
	"C:\Qt\6.9.0\mingw_64\bin\windeployqt.exe"
)

$windeployqt = Get-Command windeployqt -ErrorAction SilentlyContinue
if ($windeployqt) {
	$windeployqtPath = $windeployqt.Source
} else {
	$windeployqtPath = $null
	foreach ($path in $windeployqtPaths) {
		if (Test-Path $path) {
			$windeployqtPath = $path
			break
		}
	}
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
		& $windeployqtPath "RayTracerGUI.exe" --no-translations --no-compiler-runtime 2>&1 | Out-Null
		if ($LASTEXITCODE -eq 0) {
			Write-Host "      => Qt dependencies deployed successfully`n" -ForegroundColor Green

			# Verify critical DLLs
			$criticalDlls = @("Qt6Core.dll", "Qt6Gui.dll", "Qt6Widgets.dll")
			$missingDlls = @()
			foreach ($dll in $criticalDlls) {
				if (-not (Test-Path $dll)) {
					$missingDlls += $dll
				}
			}

			if ($missingDlls.Count -gt 0) {
				Write-Host "      [WARNING] Some Qt DLLs are still missing:" -ForegroundColor Yellow
				foreach ($dll in $missingDlls) {
					Write-Host "        - $dll" -ForegroundColor Yellow
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
