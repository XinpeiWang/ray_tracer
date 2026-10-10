#!/bin/sh
# CTest driver for metal_poc_subsurface_sky (see CMakeLists.txt): a subsurface ball under a uniform white sky, given as a constant and as a white picture. Metal's mean must match
# the CPU renderer's for each, and Metal's two skies must agree with each other. Prints SUBSURFACE_SKY_TEST_OK on success.
# usage: metal_poc_subsurface_sky_test.sh <ray_tracer> <metal_poc_media_lights_check> <scratch-dir>
RT="$1"; CHK="$2"; DIR="$3"
status=0
for kind in constant image; do
  scene="gpu/metal/test_scenes/subsurface-sky-$kind.pbrt"
  "$RT" --cpu --seed 3 --output "$DIR/sss_${kind}_cpu.exr" 48 64 30 "$scene" >/dev/null 2>&1 || { echo "$kind: CPU render failed"; exit 1; }
  "$RT" --gpu --seed 3 --output "$DIR/sss_${kind}_metal.exr" 48 64 30 "$scene" >/dev/null 2>&1 || { echo "$kind: Metal render failed"; exit 1; }
  "$CHK" "sky $kind (Metal vs CPU)" "$DIR/sss_${kind}_cpu.exr" "$DIR/sss_${kind}_metal.exr" 5 || status=1
done
"$CHK" "constant vs image sky (Metal)" "$DIR/sss_image_metal.exr" "$DIR/sss_constant_metal.exr" 8 || status=1
[ $status -eq 0 ] && echo SUBSURFACE_SKY_TEST_OK
exit $status
