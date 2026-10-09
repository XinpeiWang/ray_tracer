#!/bin/sh
# CTest driver for metal_poc_cloud_furnace (see CMakeLists.txt): renders pbrt_scenes/cloud-furnace.pbrt (a non-absorbing cloud under a uniform
# sky: it must be invisible) through the real `ray_tracer --gpu` path and runs metal_poc_furnace_check. Prints FURNACE_TEST_OK on success.
# usage: metal_poc_furnace_test.sh <ray_tracer> <metal_poc_furnace_check> <scratch-dir>
RT="$1"; CHK="$2"; DIR="$3"
METAL_BANDS=16 "$RT" --gpu --seed 5 --output "$DIR/furnace_test.png" 300 64 40 pbrt_scenes/cloud-furnace.pbrt >/dev/null 2>&1 || { echo "render failed"; exit 1; }
"$CHK" "$DIR/furnace_test.png" 2 || exit 1
echo FURNACE_TEST_OK
