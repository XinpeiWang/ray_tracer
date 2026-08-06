param([string]$Config = 'Release', [switch]$Rebuild)
$RepoRoot   = $PSScriptRoot
$SlnPath    = Join-Path $RepoRoot 'ray_tracer.sln'
$QtBuildDir = Join-Path $RepoRoot 'qt_gui'
$MinGWBin   = 'C:\Qt\Tools\mingw1310_64\bin'
$PackageDir = Join-Path $RepoRoot 'RayTracer_Package'
function OK   { param($m) Write-Host "  [OK]  $m" -ForegroundColor Green  }
function FAIL { param($m) Write-Host "  [!!]  $m" -ForegroundColor Red    }
function INFO { param($m) Write-Host "  [..]  $m" -ForegroundColor Cyan   }
function HDR  { param($m) Write-Host ''; Write-Host "=== $m ===" -ForegroundColor Yellow }
HDR 'Step 1/4  MinGW PATH'
if (-not (Test-Path $MinGWBin)) { FAIL "MinGW not found at $MinGWBin"; exit 1 }
$env:PATH = "$MinGWBin;" + $env:PATH
OK 'MinGW added'
HDR 'Step 2/4  Stop GUI'
$gui = Get-Process -Name 'RayTracerGUI' -ErrorAction SilentlyContinue
if ($gui) {
    INFO 'Stopping RayTracerGUI.exe...'
    Stop-Process -Name 'RayTracerGUI' -Force
    Start-Sleep -Milliseconds 800
    OK 'Stopped'
} else { OK 'GUI not running' }
HDR "Step 3/4  MSBuild ($Config)"
$msArgs = @($SlnPath, "/p:Configuration=$Config", '/p:Platform=x64', '/v:minimal', "/m:$([System.Environment]::ProcessorCount)")
if ($Rebuild) { $msArgs += '/t:Rebuild' }
$out = & msbuild @msArgs 2>&1
$errs  = $out | Where-Object { $_ -match ' error C\d|: error :|Build FAILED' }
$warns = $out | Where-Object { $_ -match ' warning C\d' }
if ($LASTEXITCODE -ne 0 -or $errs.Count -gt 0) {
    FAIL 'MSBuild FAILED'
    $errs | ForEach-Object { Write-Host "       $_" -ForegroundColor Red }
    exit 1
}
if ($warns.Count -gt 0) { INFO "$($warns.Count) warning(s)" }
OK 'MSBuild succeeded'
HDR 'Step 4/4  Qt GUI (MinGW)'
if (-not (Test-Path $QtBuildDir)) { FAIL "Qt build dir not found: $QtBuildDir"; exit 1 }
Push-Location $QtBuildDir
try {
    $mf = "Makefile.$Config"
    if ($Rebuild) { & mingw32-make -f $mf clean 2>&1 | Out-Null }
    $mo = & mingw32-make -f $mf 2>&1
    $me = $mo | Where-Object { $_ -match 'error:|Error \d' }
    if ($LASTEXITCODE -ne 0 -or $me.Count -gt 0) {
        FAIL 'mingw32-make FAILED'
        $me | ForEach-Object { Write-Host "       $_" -ForegroundColor Red }
        exit 1
    }
    OK 'Qt GUI built'
} finally { Pop-Location }
HDR 'Summary'
foreach ($fname in @('ray_tracer.exe','RayTracerGUI.exe','optix_programs.ptx')) {
    $full = Join-Path $PackageDir $fname
    if (Test-Path $full) {
        $t = (Get-Item $full).LastWriteTime.ToString('HH:mm:ss')
        OK "$fname  (updated $t)"
    } else { FAIL "$fname  NOT found in RayTracer_Package" }
}
Write-Host ''
Write-Host '  All done! Launch: RayTracer_Package\RayTracerGUI.exe' -ForegroundColor Green
