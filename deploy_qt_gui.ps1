# Deploy Ray Tracer Qt GUI with all dependencies
# This script copies the console launcher, PTX shader, and Qt dependencies to the package directory

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

# Find launcher executable (try multiple locations)
$launcherPaths = @(
	"$repoRoot\x64\$Configuration\ray_tracer.exe",
	"$repoRoot\launcher\x64\$Configuration\ray_tracer.exe"
)

$launcherExe = $null
foreach ($path in $launcherPaths) {
	if (Test-Path $path) {
		$launcherExe = $path
		break
	}
}

if (-not $launcherExe) {
	Write-Host "[ERROR] Launcher executable not found in any of these locations:" -ForegroundColor Red
	foreach ($path in $launcherPaths) {
		Write-Host "  - $path" -ForegroundColor Yellow
	}
	Write-Host "`nPlease build the launcher project first:" -ForegroundColor Yellow
	Write-Host "  msbuild launcher\launcher.vcxproj /p:Configuration=$Configuration /p:Platform=x64" -ForegroundColor Yellow
	exit 1
}

# Find PTX file
$ptxFile = "$repoRoot\gpu\optix\optix_programs.ptx"
if (-not (Test-Path $ptxFile)) {
	Write-Host "[ERROR] PTX shader file not found: $ptxFile" -ForegroundColor Red
	Write-Host "Please build the optix_renderer project first." -ForegroundColor Yellow
	exit 1
}

if (-not (Test-Path $ptxFile)) {
	Write-Host "[ERROR] PTX shader file not found: $ptxFile" -ForegroundColor Red
	Write-Host "Please build the optix_renderer project first." -ForegroundColor Yellow
	exit 1
}

if (-not (Test-Path $packageDir)) {
	New-Item -ItemType Directory -Path $packageDir | Out-Null
	Write-Host "[INFO] Created package directory: $packageDir`n" -ForegroundColor Green
}

# Copy files
Write-Host "[Step 1/4] Copying console launcher..." -ForegroundColor Cyan
Copy-Item $launcherExe "$packageDir\ray_tracer.exe" -Force
$launcherInfo = Get-Item "$packageDir\ray_tracer.exe"
Write-Host "      => ray_tracer.exe ($([math]::Round($launcherInfo.Length/1KB)) KB, $($launcherInfo.LastWriteTime))`n" -ForegroundColor Gray

# Also copy to other locations for consistency
Copy-Item $launcherExe "$repoRoot\qt_gui\ray_tracer.exe" -Force -ErrorAction SilentlyContinue
Copy-Item $launcherExe "$repoRoot\x64\$Configuration\ray_tracer.exe" -Force -ErrorAction SilentlyContinue
Write-Host "      (Also copied to qt_gui\ and x64\$Configuration\ for consistency)`n" -ForegroundColor DarkGray

Write-Host "[Step 2/4] Copying OptiX shader (PTX)..." -ForegroundColor Cyan
Copy-Item $ptxFile "$packageDir\optix_programs.ptx" -Force
$ptxInfo = Get-Item "$packageDir\optix_programs.ptx"
Write-Host "      => optix_programs.ptx ($([math]::Round($ptxInfo.Length/1KB)) KB, $($ptxInfo.LastWriteTime))`n" -ForegroundColor Gray

Write-Host "[Step 3/4] Copying Qt GUI executable..." -ForegroundColor Cyan
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

Write-Host "[Step 4/4] Deploying Qt dependencies with windeployqt..." -ForegroundColor Cyan
Write-Host "[Step 4/4] Deploying Qt dependencies with windeployqt..." -ForegroundColor Cyan

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
