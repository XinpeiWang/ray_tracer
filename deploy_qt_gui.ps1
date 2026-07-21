# Deploy Ray Tracer Qt GUI with all dependencies
# This script copies the console launcher and PTX shader to the Qt GUI package directory

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Ray Tracer Qt GUI Deployment Script" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

$repoRoot = $PSScriptRoot
$launcherExe = "$repoRoot\launcher\x64\Release\ray_tracer.exe"
$ptxFile = "$repoRoot\gpu\optix\optix_programs.ptx"
$packageDir = "$repoRoot\RayTracer_Package"

# Check if files exist
if (-not (Test-Path $launcherExe)) {
	Write-Host "[ERROR] Launcher executable not found: $launcherExe" -ForegroundColor Red
	Write-Host "Please build the launcher project first:" -ForegroundColor Yellow
	Write-Host "  msbuild launcher\launcher.vcxproj /p:Configuration=Release /p:Platform=x64" -ForegroundColor Yellow
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
Write-Host "[1/3] Copying console launcher..." -ForegroundColor Cyan
Copy-Item $launcherExe "$packageDir\ray_tracer.exe" -Force
$launcherInfo = Get-Item "$packageDir\ray_tracer.exe"
Write-Host "      => ray_tracer.exe ($([math]::Round($launcherInfo.Length/1KB)) KB, $($launcherInfo.LastWriteTime))`n" -ForegroundColor Gray

Write-Host "[2/3] Copying OptiX shader (PTX)..." -ForegroundColor Cyan
Copy-Item $ptxFile "$packageDir\optix_programs.ptx" -Force
$ptxInfo = Get-Item "$packageDir\optix_programs.ptx"
Write-Host "      => optix_programs.ptx ($([math]::Round($ptxInfo.Length/1KB)) KB, $($ptxInfo.LastWriteTime))`n" -ForegroundColor Gray

Write-Host "[3/3] Verifying Qt dependencies..." -ForegroundColor Cyan
$qtGuiExe = "$packageDir\RayTracerGUI.exe"
if (-not (Test-Path $qtGuiExe)) {
	Write-Host "      [WARNING] Qt GUI executable not found: $qtGuiExe" -ForegroundColor Yellow
	Write-Host "      You need to build the Qt GUI first using qmake/Qt Creator.`n" -ForegroundColor Yellow
} else {
	$qt6Core = "$packageDir\Qt6Core.dll"
	if (-not (Test-Path $qt6Core)) {
		Write-Host "      [WARNING] Qt6 DLLs not found. Run windeployqt:" -ForegroundColor Yellow
		Write-Host "      windeployqt.exe RayTracerGUI.exe --no-translations`n" -ForegroundColor Yellow
	} else {
		Write-Host "      => Qt6 dependencies present`n" -ForegroundColor Gray
	}
}

Write-Host "========================================" -ForegroundColor Green
Write-Host "Deployment Complete!" -ForegroundColor Green
Write-Host "========================================`n" -ForegroundColor Green

Write-Host "To launch the Qt GUI:" -ForegroundColor Cyan
Write-Host "  Start-Process `"$packageDir\RayTracerGUI.exe`"`n" -ForegroundColor White

Write-Host "Files deployed to: $packageDir" -ForegroundColor Gray
