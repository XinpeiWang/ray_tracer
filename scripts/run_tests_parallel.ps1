#!/usr/bin/env pwsh
<#
.SYNOPSIS
	Run ray_tracer_tests.exe sharded across multiple parallel processes.
.DESCRIPTION
	Google Test has built-in sharding support (GTEST_TOTAL_SHARDS/
	GTEST_SHARD_INDEX env vars - no test-side code changes needed): each
	shard runs a disjoint ~1/N slice of the full suite. This script launches
	one process per shard concurrently and aggregates the results.

	Correctness caveat this script exists to handle: naively running
	multiple ray_tracer_tests.exe processes at once is NOT safe out of the
	box. Some tests write output through a fixed relative filename (e.g.
	specparity_B3_spec.ppm) with no per-process uniqueness, so two shards
	racing on the same filename corrupt each other's output; other tests
	(e.g. pbrt_example_scenes_tests.cpp) scan pbrt_scenes/ by a relative
	path to build their parameterized instantiation list. This script gives
	each shard its own scratch working directory (fixing the first problem)
	with a directory JUNCTION back to every real top-level project
	directory (fixing the second, since junctions are reads-through, not
	copies) - verified to reproduce the exact same pass count as a plain
	serial run (4009/4009 passed, 0 failed, arithmetic-exact against total
	tests ran), where an earlier, working-directory-naive version of this
	script silently lost ~150 tests to the second problem and produced
	spurious OptiX/render failures from the first.

	PERFORMANCE CAVEAT - read before assuming this makes the full suite
	faster: it measurably does NOT, for the full suite as it stands today.
	Measured on this project's own machine: 2 shards of ~half the suite
	each took ~230s wall-clock, slower than the ~215s a single serial
	process takes for the WHOLE suite. Root causes: (1) many render/
	integration tests already spawn their own worker-thread pool sized to
	every logical core (determine_render_thread_count(), src/
	TheRestOfYourLife/thread_count.h) - this script caps each shard's pool
	via that file's own RAY_TRACER_THREADS override to reduce (not
	eliminate) oversubscription; (2) a handful of very expensive GPU-
	touching integration/parity tests dominate the suite's total runtime,
	contend for the ONE shared physical GPU regardless of CPU thread caps,
	and aren't evenly distributed by gtest's count-based (not time-based)
	sharding, so wall-clock ends up bounded by whichever shard drew the
	unlucky slow tests rather than shrinking with shard count. This script
	is still useful for sharding a FAST, GPU-free subset via -Filter during
	dev iteration (no shared-GPU contention there), and its two correctness
	fixes above are real and worth keeping regardless - just don't reach
	for -Shards on the full suite expecting a full-suite speedup.

	Requires the tests project to already be built (see build_all.ps1 /
	setup_env.bat) - this script only RUNS the existing exe, it doesn't
	build it.
.PARAMETER Configuration
	Build configuration whose test exe to run: Debug or Release (default: Release)
.PARAMETER Shards
	Number of parallel shards (default: logical processor count)
.PARAMETER Filter
	Optional --gtest_filter pattern, applied identically within every shard
	(each shard still only runs its own slice of whatever matches)
.EXAMPLE
	.\run_tests_parallel.ps1
	.\run_tests_parallel.ps1 -Shards 8
	.\run_tests_parallel.ps1 -Filter "CameraMathTest.*"
#>

param(
	[ValidateSet("Debug", "Release")]
	[string]$Configuration = "Release",

	[int]$Shards = [System.Environment]::ProcessorCount,

	[string]$Filter = "",

	# Generous default (the full serial suite alone takes ~215s) - exists so
	# one hung shard can't block the script forever, not to tightly bound
	# normal runs.
	[int]$TimeoutSeconds = 1800
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

$testsExe = Join-Path $PSScriptRoot "..\bin\$Configuration\ray_tracer_tests.exe"
if (-not (Test-Path $testsExe)) {
	Write-Host "[FAIL] Tests executable not found: $testsExe" -ForegroundColor Red
	Write-Host "Build it first (see scripts\build_all.ps1 or the full ray_tracer.sln)."
	exit 1
}
$testsExe = (Resolve-Path $testsExe).Path

Write-Host "Running $Shards shards of $testsExe" -ForegroundColor Cyan
if ($Filter) { Write-Host "Filter: $Filter" -ForegroundColor Cyan }

# $PID-namespaced: without this, two concurrent invocations of this script
# (two terminals, overlapping CI runs) would reuse the exact same cwd_N/
# shard_N.log paths and race on them - precisely the cross-process
# collision this script's own per-shard isolation exists to prevent, just
# one level up (between invocations instead of between shards).
$logDir = Join-Path $env:TEMP "ray_tracer_test_shards_$PID"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

# Computed once and reused for every shard below (was re-enumerated inside
# the loop, redoing the same filesystem walk once per shard for no reason -
# the repo root's own top-level directory listing doesn't change between
# shards).
$repoTopLevelDirs = Get-ChildItem -Path $repoRoot -Directory

$jobs = for ($i = 0; $i -lt $Shards; $i++) {
	$logFile = Join-Path $logDir "shard_$i.log"
	# Several tests write output through a FIXED relative filename (e.g.
	# specparity_B3_spec.ppm) with no per-process uniqueness of their own -
	# fine for a single process, but two shards racing on the SAME filename
	# in the SAME working directory corrupt each other's output and fail
	# with things like "render failed to produce a valid PPM" that have
	# nothing to do with a real bug. Each shard gets its own scratch working
	# directory so relative paths can never collide between shards. Several
	# OTHER tests read fixed INPUT resources by a relative path instead
	# (e.g. pbrt_example_scenes_tests.cpp scans pbrt_scenes/ for its
	# parameterized instantiation list) - an empty scratch directory would
	# make those find nothing and silently register far fewer test
	# instances than the real suite has. Junctioning (not copying) every
	# top-level directory from the real repo root into the scratch
	# directory keeps all such reads resolving exactly as they do from the
	# real root, while a bare-filename WRITE (nothing this project does
	# writes through a subdirectory) still lands fresh in the scratch
	# directory itself, never touching the shared original.
	$shardDir = Join-Path $logDir "cwd_$i"
	New-Item -ItemType Directory -Force -Path $shardDir | Out-Null
	foreach ($dir in $repoTopLevelDirs) {
		New-Item -ItemType Junction -Path (Join-Path $shardDir $dir.Name) -Target $dir.FullName -Force | Out-Null
	}
	# Many render/integration tests spawn their OWN worker-thread pool sized
	# to the full logical-core count (determine_render_thread_count(),
	# src/TheRestOfYourLife/thread_count.h) - fine for one process, but N
	# concurrent shards each independently claiming every core oversubscribes
	# the machine and makes the WHOLE run slower than running serially in one
	# process (measured: 2 shards of ~half the suite each took LONGER wall-
	# clock than the full suite in one process). RAY_TRACER_THREADS is that
	# same file's own explicit override - capping each shard to its fair
	# share of the machine (rounded up so the last shard doesn't starve on a
	# core count that doesn't divide evenly) keeps total concurrent threads
	# across all shards close to the real core count instead of Shards-times
	# over it.
	$threadsPerShard = [Math]::Max(1, [Math]::Ceiling([System.Environment]::ProcessorCount / $Shards))
	Start-Job -Name "shard_$i" -ArgumentList $testsExe, $Shards, $i, $Filter, $logFile, $shardDir, $threadsPerShard -ScriptBlock {
		param($exe, $total, $index, $filter, $log, $cwd, $threads)
		Set-Location $cwd
		$env:GTEST_TOTAL_SHARDS = $total
		$env:GTEST_SHARD_INDEX = $index
		$env:RAY_TRACER_THREADS = $threads
		$argList = @("--gtest_brief=1")
		if ($filter) { $argList += "--gtest_filter=$filter" }
		& $exe @argList *> $log
		# Return (not `exit`) the real process exit code so the parent can
		# read it via Receive-Job. `exit` inside a Start-Job scriptblock only
		# terminates that job's own runspace - it is NOT reflected in
		# $job.State (State stays "Completed" even after `exit 1`), so a
		# hard crash (access violation, etc.) used to look identical to a
		# clean run to every check the parent script had.
		$LASTEXITCODE
	}
}

Write-Host "Waiting for $($jobs.Count) shards (timeout: ${TimeoutSeconds}s)..."
Wait-Job -Job $jobs -Timeout $TimeoutSeconds | Out-Null

# Wait-Job's own -Timeout only stops WAITING - a job still Running after it
# elapses keeps executing in the background forever unless explicitly
# stopped. Without this, a single deadlocked/hung test (GPU driver stall,
# etc.) in any one shard used to block the whole script indefinitely with
# no diagnostic at all.
$hungJobs = $jobs | Where-Object { $_.State -eq "Running" }
foreach ($job in $hungJobs) {
	Write-Host "[FAIL] $($job.Name): timed out after ${TimeoutSeconds}s - stopping" -ForegroundColor Red
}
if ($hungJobs) { Stop-Job -Job $hungJobs | Out-Null }

$failedShards = @()
$totalPassed = 0
$totalFailed = 0
foreach ($job in $jobs) {
	$logFile = Join-Path $logDir "$($job.Name).log"
	$content = if (Test-Path $logFile) { Get-Content $logFile -Raw } else { "" }
	# Receive-Job returns the scriptblock's own last expression (see the job
	# definition above) - the real process exit code, not PowerShell's
	# notion of job State (which stays "Completed" even after a crash).
	$exitCode = Receive-Job -Job $job -ErrorAction SilentlyContinue
	$passedMatch = [regex]::Match($content, '\[\s*PASSED\s*\]\s*(\d+)')
	$failedMatch = [regex]::Match($content, '\[\s*FAILED\s*\]\s*(\d+)')
	$passedCount = if ($passedMatch.Success) { [int]$passedMatch.Groups[1].Value } else { 0 }
	$failedCount = if ($failedMatch.Success) { [int]$failedMatch.Groups[1].Value } else { 0 }
	$totalPassed += $passedCount
	$totalFailed += $failedCount

	# A crash (or a timeout kill above) before gtest ever prints its own
	# summary line looks identical to "0 passed, 0 failed" to the two
	# regexes above - genuinely distinguishing that from a real 0-test shard
	# (e.g. an over-narrow -Filter) needs both the real exit code AND
	# whether a PASSED line ever appeared at all, neither of which the old
	# $job.State/text-search checks here actually caught.
	$crashedOrHung = ($job.State -eq "Stopped") -or ($null -ne $exitCode -and $exitCode -ne 0) -or (-not $passedMatch.Success)
	if ($crashedOrHung -or $failedCount -gt 0) {
		$failedShards += $job.Name
		$reason = if ($crashedOrHung) { "crashed/timed out (exit code: $exitCode)" } else { "$failedCount test(s) failed" }
		Write-Host "[FAIL] $($job.Name): $passedCount passed, $failedCount failed - $reason - see $logFile" -ForegroundColor Red
	} else {
		Write-Host "[OK] $($job.Name): $passedCount passed" -ForegroundColor Green
	}
}

Remove-Job -Job $jobs

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Total: $totalPassed passed, $totalFailed failed across $Shards shards" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

if ($failedShards.Count -gt 0) {
	Write-Host "Failed shards: $($failedShards -join ', ')" -ForegroundColor Red
	Write-Host "Full logs kept in: $logDir"
	exit 1
}

Write-Host "All shards passed." -ForegroundColor Green
exit 0
