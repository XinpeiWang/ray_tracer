#!/bin/sh
# CTest driver for metal_poc_alpha_cutout (see CMakeLists.txt): renders pbrt_scenes/alpha-cutout.pbrt (a square with a "texture alpha"
# cutout mask in front of a wall) through the real `ray_tracer --gpu` path and runs metal_poc_alpha_check on the picture.
# Prints ALPHA_TEST_OK only if the checker passes.
# usage: metal_poc_alpha_test.sh <ray_tracer> <metal_poc_alpha_check> <scratch-dir>
RT="$1"; CHK="$2"; DIR="$3"
"$RT" --gpu --seed 5 --output "$DIR/alpha_test.png" 128 32 4 pbrt_scenes/alpha-cutout.pbrt >/dev/null 2>&1 || { echo "render failed"; exit 1; }
"$CHK" "$DIR/alpha_test.png" || exit 1
echo ALPHA_TEST_OK
