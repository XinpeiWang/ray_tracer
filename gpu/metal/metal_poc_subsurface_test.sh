#!/bin/sh
# CTest driver for metal_poc_subsurface (see CMakeLists.txt): renders pbrt_scenes/subsurface-ball.pbrt (a Cornell box with two balls of pbrt's
# SubsurfaceMaterial) through the real `ray_tracer --gpu` path and runs metal_poc_subsurface_check on the picture. Prints SUBSURFACE_TEST_OK on success.
# usage: metal_poc_subsurface_test.sh <ray_tracer> <metal_poc_subsurface_check> <scratch-dir>
RT="$1"; CHK="$2"; DIR="$3"
"$RT" --gpu --seed 5 --output "$DIR/subsurface_test.png" 128 256 8 pbrt_scenes/subsurface-ball.pbrt >/dev/null 2>&1 || { echo "render failed"; exit 1; }
"$CHK" "$DIR/subsurface_test.png" 8 || exit 1
echo SUBSURFACE_TEST_OK
