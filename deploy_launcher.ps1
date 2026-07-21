# Verify Launcher Deployment
# Checks that ray_tracer.exe and optix_programs.ptx are correctly deployed to RayTracer_Package
# MSBuild post-build events should auto-deploy; this script verifies it worked

$RepoRoot = $PSScriptRoot
$PackageDir = Join-Path $RepoRoot "RayTracer_Package"
$ExpectedExe = Join-Path $PackageDir "ray_tracer.exe"
$ExpectedPTX = Join-Path $PackageDir "optix_programs.ptx"

Write-Host "Verifying deployment to $PackageDir..." -ForegroundColor Cyan

# Check if package directory exists
if (-not (Test-Path $PackageDir)) {
	Write-Error "Package directory not found: $PackageDir"
	Write-Host "└─ Build the launcher project first to create it automatically."
	exit 1
}

# Check executable
if (Test-Path $ExpectedExe) {
	$exeTime = (Get-Item $ExpectedExe).LastWriteTime
	Write-Host "✓ ray_tracer.exe found (updated: $exeTime)" -ForegroundColor Green
} else {
	Write-Error "✗ ray_tracer.exe NOT FOUND"
	Write-Host "└─ Build launcher project: msbuild launcher\launcher.vcxproj /p:Configuration=Release /p:Platform=x64"
	exit 1
}

# Check PTX
if (Test-Path $ExpectedPTX) {
	$ptxTime = (Get-Item $ExpectedPTX).LastWriteTime
	Write-Host "✓ optix_programs.ptx found (updated: $ptxTime)" -ForegroundColor Green
} else {
	Write-Error "✗ optix_programs.ptx NOT FOUND"
	Write-Host "└─ Build optix_renderer project: msbuild optix_renderer\optix_renderer.vcxproj /p:Configuration=Release /p:Platform=x64"
	exit 1
}

Write-Host "`n✅ Deployment verified! The GUI will use these files." -ForegroundColor Green
Write-Host "`nℹ️  Note: MSBuild automatically deploys after each build via post-build events." -ForegroundColor Cyan
Write-Host "   You don't need to manually copy files. Just rebuild the projects when you change code." -ForegroundColor Cyan
