#!/bin/sh
# CTest driver for metal_poc_crop_window (see CMakeLists.txt): renders scene
# A1 twice through the real `ray_tracer --gpu` path - once whole, once with
# --crop 0.25 0.25 0.75 0.75 - then runs metal_poc_crop_check on the pair.
# At 128x128 that window resolves to pixels [32,96) x [32,96) (the shared
# lround()-based resolver, src/shared/cameras.h). Prints CROP_TEST_OK only if
# the checker passes.
# usage: metal_poc_crop_test.sh <ray_tracer> <metal_poc_crop_check> <scratch-dir>
RT="$1"; CHK="$2"; DIR="$3"
"$RT" --gpu --output "$DIR/crop_test_full.png" 128 8 3 A1 >/dev/null 2>&1 || { echo "full render failed"; exit 1; }
"$RT" --gpu --crop 0.25 0.25 0.75 0.75 --output "$DIR/crop_test_crop.png" 128 8 3 A1 >/dev/null 2>&1 || { echo "crop render failed"; exit 1; }
"$CHK" "$DIR/crop_test_full.png" "$DIR/crop_test_crop.png" 32 96 32 96 8 || exit 1
echo CROP_TEST_OK
