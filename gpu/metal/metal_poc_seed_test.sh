#!/bin/sh
# CTest driver for metal_poc_seed_reproducible (see CMakeLists.txt): renders
# scene A1 through the real `ray_tracer --gpu` path and checks the three
# properties --seed must have under Metal (docs/METAL_GPU_FEASIBILITY.md
# section 204): the same seed reproduces the image exactly, a different seed
# gives a different image, and an explicit seed differs from passing none.
# Prints SEED_TEST_OK only if all three hold (the test's
# PASS_REGULAR_EXPRESSION), otherwise says which one failed.
# usage: metal_poc_seed_test.sh <path-to-ray_tracer> <scratch-dir>
RT="$1"; DIR="$2"
render() { "$RT" --gpu $1 --output "$DIR/seed_test_$2.png" 64 4 2 A1 >/dev/null 2>&1 || { echo "render failed ($1)"; exit 1; }
           shasum -a 256 "$DIR/seed_test_$2.png" | cut -d' ' -f1; }
A=$(render "--seed 5" a); B=$(render "--seed 5" b); C=$(render "--seed 6" c); D=$(render "" d)
[ -n "$A" ] && [ "$A" = "$B" ] || { echo "same seed did NOT reproduce"; exit 1; }
[ "$A" != "$C" ] || { echo "different seeds gave identical images"; exit 1; }
[ "$A" != "$D" ] || { echo "explicit seed identical to no seed"; exit 1; }
echo SEED_TEST_OK
