#!/bin/sh
# CTest driver for metal_poc_sppm_gpu_falls_back (see CMakeLists.txt): on a Mac `ray_tracer --sppm --gpu` has no Metal SPPM to run, so like --bdpt/--mlt it must
# warn that --gpu is ignored and render on the CPU (docs/BACKEND_SUPPORT.md: SPPM is "CPU" under Metal). It used to stop with "OptiX is not available!".
# Prints SPPM_FALLBACK_OK only if it exits 0, says so, and writes the picture.
# usage: metal_poc_sppm_fallback_test.sh <path-to-ray_tracer> <scratch-dir>
RT="$1"; DIR="$2"
OUT="$DIR/sppm_fallback_test.png"
rm -f "$OUT"
LOG=$("$RT" --sppm --gpu --sppm-iterations 4 --sppm-photons 500 --output "$OUT" 48 4 3 A1 2>&1); RC=$?
[ "$RC" = 0 ] || { echo "exit code $RC; log:"; echo "$LOG" | tail -5; exit 1; }
echo "$LOG" | grep -q -- "--gpu is ignored under --sppm" || { echo "no 'ignored' warning; log:"; echo "$LOG" | tail -5; exit 1; }
[ -s "$OUT" ] || { echo "no picture written"; exit 1; }
rm -f "$OUT" "$OUT.run_marker.txt"
echo SPPM_FALLBACK_OK
