# RayTracer Packaging Script
# Collects all dependencies and creates a distributable package

param(
	[string]$Config = "Release",
	[string]$OutputDir = ".\RayTracer_Package"
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Ray Tracer Packaging Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Paths
$ProjectRoot = $PSScriptRoot
$BuildDir = Join-Path $ProjectRoot "x64\$Config"
$OutputDir = Join-Path $ProjectRoot $OutputDir

# Clean and create output directory
Write-Host "`nPreparing output directory: $OutputDir"
if (Test-Path $OutputDir) {
	Remove-Item -Recurse -Force $OutputDir
}
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
New-Item -ItemType Directory -Path "$OutputDir\output" -Force | Out-Null

# Copy main executable
Write-Host "`nCopying main executable..."
Copy-Item "$BuildDir\ray_tracer.exe" "$OutputDir\RayTracer.exe" -Force
if (-not (Test-Path "$OutputDir\RayTracer.exe")) {
	Write-Host "ERROR: Could not find ray_tracer.exe in $BuildDir" -ForegroundColor Red
	exit 1
}
Write-Host "OK RayTracer.exe" -ForegroundColor Green

# Copy GUI executable if it exists
if (Test-Path "$BuildDir\RayTracerGUI.exe") {
	Copy-Item "$BuildDir\RayTracerGUI.exe" "$OutputDir\RayTracerGUI.exe" -Force
	Write-Host "OK RayTracerGUI.exe" -ForegroundColor Green
}

# Copy launcher batch file
if (Test-Path "$ProjectRoot\launcher.bat") {
	Copy-Item "$ProjectRoot\launcher.bat" "$OutputDir\launcher.bat" -Force
	Write-Host "OK launcher.bat" -ForegroundColor Green
}

# Copy README
if (Test-Path "$ProjectRoot\README_PACKAGE.txt") {
	Copy-Item "$ProjectRoot\README_PACKAGE.txt" "$OutputDir\README.txt" -Force
	Write-Host "OK README.txt" -ForegroundColor Green
}

# Find and copy CUDA runtime DLLs
Write-Host "`nSearching for CUDA runtime DLLs..."
$cudaPaths = @(
	"$env:CUDA_PATH\bin",
	"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin",
	"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin",
	"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"
)

$cudaDlls = @("cudart64_133.dll", "cudart64_132.dll", "cudart64_126.dll")
$foundCuda = $false

foreach ($path in $cudaPaths) {
	if (Test-Path $path) {
		foreach ($dll in $cudaDlls) {
			$dllPath = Join-Path $path $dll
			if (Test-Path $dllPath) {
				Copy-Item $dllPath $OutputDir -Force
				Write-Host "OK $dll" -ForegroundColor Green
				$foundCuda = $true
				break
			}
		}
		if ($foundCuda) { break }
	}
}

if (-not $foundCuda) {
	Write-Host "WARNING: CUDA runtime not found - GPU mode will not work" -ForegroundColor Yellow
}

# Find and copy Visual C++ Runtime DLLs
Write-Host "`nSearching for Visual C++ Runtime DLLs..."
$vcRedistPaths = @(
	"C:\Program Files\Microsoft Visual Studio\2026\Community\VC\Redist\MSVC\14.50.34831\x64\Microsoft.VC145.CRT",
	"C:\Windows\System32"
)

$vcDlls = @("vcruntime140_1.dll", "vcruntime140.dll", "msvcp140.dll")
foreach ($vcDll in $vcDlls) {
	$found = $false
	foreach ($path in $vcRedistPaths) {
		$dllPath = Join-Path $path $vcDll
		if (Test-Path $dllPath) {
			Copy-Item $dllPath $OutputDir -Force -ErrorAction SilentlyContinue
			Write-Host "OK $vcDll" -ForegroundColor Green
			$found = $true
			break
		}
	}
	if (-not $found) {
		# Try to find in system PATH
		$systemDll = (Get-Command $vcDll -ErrorAction SilentlyContinue).Path
		if ($systemDll) {
			Copy-Item $systemDll $OutputDir -Force
			Write-Host "OK $vcDll (from system)" -ForegroundColor Green
		}
	}
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Package created successfully!" -ForegroundColor Green
Write-Host "Location: $OutputDir" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "`nTo test the package:"
Write-Host "  cd $OutputDir"
Write-Host "  .\launcher.bat"
Write-Host "`nTo create ZIP for distribution:"
Write-Host "  Compress-Archive -Path $OutputDir -DestinationPath RayTracer_Portable.zip"
