#!/bin/sh
# CTest driver for metal_poc_nanovdb (see CMakeLists.txt): renders the two NanoVDB check scenes through the real `ray_tracer --gpu` path and runs
# metal_poc_nanovdb_check on the pictures. Prints NANOVDB_TEST_OK on success.
# usage: metal_poc_nanovdb_test.sh <ray_tracer> <metal_poc_nanovdb_check> <scratch-dir>
RT="$1"; CHK="$2"; DIR="$3"
"$RT" --gpu --seed 5 --output "$DIR/nanovdb_furnace.exr" 128 40 8 pbrt_scenes/nanovdb-furnace.pbrt >/dev/null 2>&1 || { echo "furnace render failed"; exit 1; }
"$RT" --gpu --seed 5 --output "$DIR/nanovdb_slab.exr" 64 256 8 pbrt_scenes/nanovdb-thin-slab.pbrt >/dev/null 2>&1 || { echo "slab render failed"; exit 1; }
"$CHK" "$DIR/nanovdb_furnace.exr" "$DIR/nanovdb_slab.exr" || exit 1
echo NANOVDB_TEST_OK
