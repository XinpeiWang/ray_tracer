#!/bin/sh
# CTest driver for metal_poc_clipped_medium (see CMakeLists.txt): a non-absorbing cloud whose box sticks out of its interface sphere (test_scenes/cloud-furnace-clipped.pbrt) under a
# uniform sky of 0.5 must be invisible. Prints CLIPPED_MEDIUM_TEST_OK on success.
# usage: metal_poc_clipped_medium_test.sh <ray_tracer> <metal_poc_furnace_blocks_check> <scratch-dir>
RT="$1"; CHK="$2"; DIR="$3"
"$RT" --gpu --seed 2 --output "$DIR/clipped_medium.exr" 150 128 100 gpu/metal/test_scenes/cloud-furnace-clipped.pbrt >/dev/null 2>&1 || { echo "render failed"; exit 1; }
"$CHK" "$DIR/clipped_medium.exr" 0.5 1.5 0.93 || exit 1
echo CLIPPED_MEDIUM_TEST_OK
