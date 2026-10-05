#!/usr/bin/env pwsh
<#
.SYNOPSIS
	Compile the project's CUDA/OptiX sources with exact header dependencies, in parallel.
.DESCRIPTION
	Called by build_optix.targets (target CompileCudaSources) instead of one MSBuild Exec per file. It replaces three weaknesses of that:

	1. Dependencies. MSBuild's Inputs/Outputs could only be given a blanket list ("every header in gpu/optix and src/shared"), so touching ANY of
	   those headers - including host-only ones no .cu includes - recompiled all 16 CUDA files (~4 minutes). Each compile here also writes
	   nvcc's own dependency file (-MD -MF); a file is rebuilt only when its source, one of the headers it actually included, or its compile
	   flags changed. No dependency file (first build, or a clean) means rebuild.
	2. Parallelism. The 16 compiles are independent (each .cu -> its own .ptx/.obj) but ran one after another; they now run -Parallel at a time.
	3. Visibility. Prints which files were rebuilt and how long each took.

	Job file: one job per line, three TAB-separated fields: source, output, nvcc flags (everything but the source/output/dependency options).
	The device-link steps (nvcc -dlink) stay MSBuild targets: they have exact Inputs/Outputs already.
.PARAMETER JobFile
	Text file of jobs (see above), written by the build_optix.targets target that calls this script.
.PARAMETER Nvcc
	Full path of nvcc.exe.
.PARAMETER DepDir
	Directory for the per-source dependency (.d) and flags (.flags) files.
.PARAMETER Parallel
	How many nvcc processes to run at once (default 6; each can use a few GB).
.PARAMETER WorkDir
	Working directory for the compiles (the repository root, as the old Exec used).
#>
param(
	[Parameter(Mandatory = $true)][string]$JobFile,
	[Parameter(Mandatory = $true)][string]$Nvcc,
	[Parameter(Mandatory = $true)][string]$DepDir,
	[int]$Parallel = 6,
	[string]$WorkDir = (Get-Location).Path
)

$ErrorActionPreference = 'Stop'
if ($Parallel -lt 1) { $Parallel = 1 }
$DepDir = [System.IO.Path]::GetFullPath($DepDir)
[System.IO.Directory]::CreateDirectory($DepDir) | Out-Null

# nvcc -MD writes a make-style file: "out : src \ <newline> dep1 \ <newline> dep2 ..." with forward slashes and "\ " for a space in a path.
function Read-Dependencies([string]$depFile) {
	$text = [System.IO.File]::ReadAllText($depFile)
	$text = $text -replace '\\\r?\n', ' '
	$tokens = [regex]::Split($text.Trim(), '(?<!\\)\s+') | Where-Object { $_ -ne '' }
	# first two tokens are "<output>" and ":"
	if ($tokens.Count -lt 3) { return @() }
	return $tokens | Select-Object -Skip 2 | ForEach-Object { ($_ -replace '\\ ', ' ') }
}

# The 16 sources share most of their ~240 headers each, so each header's timestamp is looked up once (.NET calls, not Get-Item: thousands of
# Get-Item calls cost seconds in Windows PowerShell and made a build with nothing to do take 6 s instead of ~1).
$script:mtimes = @{}
function Get-MTimeTicks([string]$path) {
	$v = $script:mtimes[$path]
	if ($null -ne $v) { return $v }
	$v = if ([System.IO.File]::Exists($path)) { [System.IO.File]::GetLastWriteTimeUtc($path).Ticks } else { [long]-1 }
	$script:mtimes[$path] = $v
	return $v
}

function Test-UpToDate($job) {
	if (-not [System.IO.File]::Exists($job.Out)) { return $false }
	if (-not [System.IO.File]::Exists($job.Dep)) { return $false }
	if (-not [System.IO.File]::Exists($job.FlagsFile)) { return $false }
	if ([System.IO.File]::ReadAllText($job.FlagsFile) -ne $job.Flags) { return $false }
	$outTicks = [System.IO.File]::GetLastWriteTimeUtc($job.Out).Ticks
	foreach ($dep in (@($job.Src) + (Read-Dependencies $job.Dep))) {
		$t = Get-MTimeTicks $dep
		if ($t -lt 0) { return $false }              # a header that no longer exists
		if ($t -gt $outTicks) { return $false }
	}
	return $true
}

$jobs = @()
foreach ($line in (Get-Content -LiteralPath $JobFile)) {
	if ([string]::IsNullOrWhiteSpace($line)) { continue }
	$f = $line -split "`t"
	if ($f.Count -ne 3) { throw "Bad job line (expected source<TAB>output<TAB>flags): $line" }
	$name = [System.IO.Path]::GetFileNameWithoutExtension($f[0])
	$jobs += [pscustomobject]@{
		Src = $f[0]; Out = $f[1]; Flags = $f[2]; Name = $name
		Dep = (Join-Path $DepDir "$name$([System.IO.Path]::GetExtension($f[1])).d")
		FlagsFile = (Join-Path $DepDir "$name$([System.IO.Path]::GetExtension($f[1])).flags")
	}
}

$todo = @($jobs | Where-Object { -not (Test-UpToDate $_) })
Write-Host "[CUDA] $($jobs.Count) source(s), $($todo.Count) to compile, $Parallel at a time"
if ($todo.Count -eq 0) { exit 0 }

$running = @()
$failed = 0
$queue = [System.Collections.Queue]::new($todo)
$total = [Diagnostics.Stopwatch]::StartNew()

function Complete-Job($r) {
	$r.Proc.WaitForExit()
	$secs = [int]((Get-Date) - $r.Start).TotalSeconds
	# ReadAllText, not Get-Content -Raw: that returns $null (not '') for an empty file, and nvcc usually prints nothing on success.
	$stdout = if (Test-Path -LiteralPath $r.StdOut) { [System.IO.File]::ReadAllText($r.StdOut) } else { '' }
	$stderr = if (Test-Path -LiteralPath $r.StdErr) { [System.IO.File]::ReadAllText($r.StdErr) } else { '' }
	if ($r.Proc.ExitCode -ne 0) {
		Write-Host "[CUDA] FAILED $($r.Job.Name) (exit $($r.Proc.ExitCode), ${secs}s)"
		if ($stdout) { Write-Host $stdout }
		if ($stderr) { Write-Host $stderr }
		$script:failed++
		# Don't leave a flags file that would make a half-built output look current.
		Remove-Item -LiteralPath $r.Job.FlagsFile -ErrorAction SilentlyContinue
	} else {
		[System.IO.File]::WriteAllText($r.Job.FlagsFile, $r.Job.Flags)
		Write-Host "[CUDA] $($r.Job.Name) -> $([System.IO.Path]::GetFileName($r.Job.Out)) (${secs}s)"
		if ($stdout.Trim()) { Write-Host $stdout }
		if ($stderr.Trim()) { Write-Host $stderr }
	}
	Remove-Item -LiteralPath $r.StdOut, $r.StdErr -ErrorAction SilentlyContinue
}

while ($queue.Count -gt 0 -or $running.Count -gt 0) {
	while ($queue.Count -gt 0 -and $running.Count -lt $Parallel -and $failed -eq 0) {
		$job = $queue.Dequeue()
		Remove-Item -LiteralPath $job.FlagsFile -ErrorAction SilentlyContinue
		$stdOut = Join-Path $DepDir "$($job.Name).$PID.out"
		$stdErr = Join-Path $DepDir "$($job.Name).$PID.err"
		$argLine = "$($job.Flags) -MD -MF `"$($job.Dep)`" `"$($job.Src)`" -o `"$($job.Out)`""
		$p = Start-Process -FilePath $Nvcc -ArgumentList $argLine -WorkingDirectory $WorkDir -NoNewWindow -PassThru `
			-RedirectStandardOutput $stdOut -RedirectStandardError $stdErr
		$null = $p.Handle    # cache the process handle so ExitCode is readable after exit
		$running += [pscustomobject]@{ Proc = $p; Job = $job; Start = (Get-Date); StdOut = $stdOut; StdErr = $stdErr }
	}
	if ($failed -gt 0) { $queue.Clear() }
	$done = @($running | Where-Object { $_.Proc.HasExited })
	foreach ($r in $done) { Complete-Job $r }
	$running = @($running | Where-Object { -not $_.Proc.HasExited })
	if ($done.Count -eq 0) { Start-Sleep -Milliseconds 200 }
}

Write-Host "[CUDA] done in $([int]$total.Elapsed.TotalSeconds)s ($($todo.Count - $failed) compiled, $failed failed)"
if ($failed -gt 0) { exit 1 }
exit 0
